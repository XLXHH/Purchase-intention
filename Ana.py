import os
import json
from datetime import datetime
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance


"""
本脚本在原 demo1.py 基础上做了增强：
1) Session 特征向量化：避免逐 visitorid 循环
2) 新增验证集 + 自动阈值选择
3) 新增 rolling time CV
4) 新增模型保存、训练历史记录、最佳模型自动比较

注意：
- 不改原有读取路径变量名
- 不改原有主要函数名
- 每次运行都会训练，并把更优结果保存为 best_model.joblib
"""


# ----------------------------
# 0) Config
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

EVENTS_PATH = os.path.join(DATA_DIR, "new.xlsx")
CATEGORY_TREE_PATH = os.path.join(DATA_DIR, "category_tree.csv")
ITEM_PROPS_PATH = os.path.join(DATA_DIR, "item_properties_part1.csv")

OBS_DAYS = 7
PRED_DAYS = 7
ANCHOR_FREQ = "7D"
MIN_EVENTS_IN_OBS = 3
'''OBS_DAYS = 0.15
PRED_DAYS = 0.15
ANCHOR_FREQ = "2H"
MIN_EVENTS_IN_OBS = 1'''


# 多尺度窗口（必须 <= OBS_DAYS）
MOMENTUM_WINDOWS_DAYS = [1, 3]

# Session 切分
SESSION_GAP_MIN = 30

# 时间切分：train / val / test
TRAIN_FRAC = 0.8
VAL_FRAC_WITHIN_TRAIN = 0.2

# 阈值选择
THRESHOLD_MODE = "max_f1"       # "max_f1" 或 "precision_at_least"
TARGET_PRECISION = 0.80

# Rolling CV
ENABLE_ROLLING_CV = True
CV_N_FOLDS = 3
CV_MIN_TRAIN_ANCHORS = 3

# ----------------------------
# Model artifact / history
# ----------------------------
ARTIFACT_DIR = os.path.join(DATA_DIR, "artifacts")
BEST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.joblib")
BEST_META_PATH = os.path.join(ARTIFACT_DIR, "best_model_meta.json")
HISTORY_PATH = os.path.join(ARTIFACT_DIR, "training_history.csv")
CANDIDATE_DIR = os.path.join(ARTIFACT_DIR, "candidates")
ROLLING_CV_PATH = os.path.join(ARTIFACT_DIR, "rolling_cv_metrics.csv")
LAST_RUN_META_PATH = os.path.join(ARTIFACT_DIR, "last_run_best_meta.csv")

PRIMARY_METRIC = "test_pr_auc"   # 可选: test_pr_auc / test_f1 / val_pr_auc


# ----------------------------
# Utils
# ----------------------------
def extract_n_values(s: str):
    """从 value 字符串中提取所有以 n 开头的数值 token，如 n277.200"""
    vals = []
    for tok in str(s).split():
        if tok.startswith("n"):
            try:
                vals.append(float(tok[1:]))
            except Exception:
                pass
    return vals


def build_models(random_state: int = 42):
    """每次调用返回一套新的模型实例。"""
    lr = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state
        ))
    ])

    hgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=random_state
    )
    return {"LogReg": lr, "HistGB": hgb}


def sanitize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """将特征列强制为数值，处理 inf/NaN，按分位数裁剪极端值。"""
    out = df.copy()

    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)

    med = out[feature_cols].median(numeric_only=True)
    out[feature_cols] = out[feature_cols].fillna(med).fillna(0)

    lower = out[feature_cols].quantile(0.001)
    upper = out[feature_cols].quantile(0.999)
    out[feature_cols] = out[feature_cols].clip(lower=lower, upper=upper, axis=1)

    if not np.isfinite(out[feature_cols].to_numpy()).all():
        bad = []
        for c in feature_cols:
            arr = out[c].to_numpy()
            if not np.isfinite(arr).all():
                bad.append(c)
        raise ValueError(f"Still contains non-finite values in columns: {bad[:50]} ... total={len(bad)}")
    return out


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                     mode: str = "max_f1", target_precision: float = 0.8):
    """
    基于 precision-recall curve 自动选择阈值：
    - max_f1：选 F1 最大的阈值
    - precision_at_least：在 precision>=target_precision 的阈值中选 recall 最大的阈值
    返回：best_thr, (precision, recall, f1)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)

    if thresholds.size == 0:
        return 0.5, (0.0, 0.0, 0.0)

    if mode == "max_f1":
        i = int(np.nanargmax(f1))
        return float(thresholds[i]), (float(precision[i]), float(recall[i]), float(f1[i]))

    if mode == "precision_at_least":
        ok = precision >= float(target_precision)
        if not ok.any():
            i = int(np.nanargmax(f1))
            return float(thresholds[i]), (float(precision[i]), float(recall[i]), float(f1[i]))
        idx = np.where(ok)[0]
        j = idx[int(np.nanargmax(recall[idx]))]
        return float(thresholds[j]), (float(precision[j]), float(recall[j]), float(f1[j]))

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_with_threshold(name: str, y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    pred = (y_prob >= thr).astype(int)

    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    f1 = f1_score(y_true, pred) if len(np.unique(y_true)) > 1 else float("nan")

    acc = accuracy_score(y_true, pred) if len(y_true) > 0 else float("nan")
    prec = precision_score(y_true, pred, zero_division=0) if len(np.unique(y_true)) > 0 else float("nan")
    rec = recall_score(y_true, pred, zero_division=0) if len(np.unique(y_true)) > 0 else float("nan")

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    correct_rate = (tp + tn) / max(len(y_true), 1)
    error_rate = (fp + fn) / max(len(y_true), 1)

    print(f"\n=== {name} (thr={thr:.4f}) ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("ROC-AUC  :", roc)
    print("PR-AUC   :", ap)
    print("F1       :", f1)
    print("Correct  :", correct_rate)
    print("Error    :", error_rate)
    print(classification_report(y_true, pred, digits=4))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
        "pr_auc": ap,
        "f1": f1,
        "correct_rate": correct_rate,
        "error_rate": error_rate,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ----------------------------
# Artifact helpers
# ----------------------------
def ensure_artifact_dirs():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(CANDIDATE_DIR, exist_ok=True)


def _to_builtin(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if pd.isna(x):
        return None
    return x


def make_run_meta(model_name: str,
                  threshold: float,
                  val_metrics: dict,
                  test_metrics: dict,
                  feature_cols: list,
                  train_df: pd.DataFrame,
                  val_df: pd.DataFrame,
                  test_df: pd.DataFrame):
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta = {
        "run_id": run_id,
        "run_time": run_time,
        "model_name": model_name,
        "threshold": float(threshold),
        "primary_metric": PRIMARY_METRIC,

        "val_accuracy": _to_builtin(val_metrics.get("accuracy")),
        "val_precision": _to_builtin(val_metrics.get("precision")),
        "val_recall": _to_builtin(val_metrics.get("recall")),
        "val_roc_auc": _to_builtin(val_metrics.get("roc_auc")),
        "val_pr_auc": _to_builtin(val_metrics.get("pr_auc")),
        "val_f1": _to_builtin(val_metrics.get("f1")),
        "val_correct_rate": _to_builtin(val_metrics.get("correct_rate")),
        "val_error_rate": _to_builtin(val_metrics.get("error_rate")),

        "test_accuracy": _to_builtin(test_metrics.get("accuracy")),
        "test_precision": _to_builtin(test_metrics.get("precision")),
        "test_recall": _to_builtin(test_metrics.get("recall")),
        "test_roc_auc": _to_builtin(test_metrics.get("roc_auc")),
        "test_pr_auc": _to_builtin(test_metrics.get("pr_auc")),
        "test_f1": _to_builtin(test_metrics.get("f1")),
        "test_correct_rate": _to_builtin(test_metrics.get("correct_rate")),
        "test_error_rate": _to_builtin(test_metrics.get("error_rate")),

        "test_tn": _to_builtin(test_metrics.get("tn")),
        "test_fp": _to_builtin(test_metrics.get("fp")),
        "test_fn": _to_builtin(test_metrics.get("fn")),
        "test_tp": _to_builtin(test_metrics.get("tp")),

        "n_features": int(len(feature_cols)),
        "feature_cols": list(feature_cols),

        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),

        "obs_days": float(OBS_DAYS),
        "pred_days": float(PRED_DAYS),
        "min_events_in_obs": int(MIN_EVENTS_IN_OBS),
        "session_gap_min": int(SESSION_GAP_MIN),
        "train_frac": float(TRAIN_FRAC),
        "val_frac_within_train": float(VAL_FRAC_WITHIN_TRAIN),
        "threshold_mode": THRESHOLD_MODE,
        "target_precision": float(TARGET_PRECISION),
        "momentum_windows_days": list(MOMENTUM_WINDOWS_DAYS),
        "enable_rolling_cv": bool(ENABLE_ROLLING_CV),
        "cv_n_folds": int(CV_N_FOLDS),
        "cv_min_train_anchors": int(CV_MIN_TRAIN_ANCHORS),
    }
    return meta


def save_candidate_artifact(model, meta: dict):
    ensure_artifact_dirs()

    run_id = meta["run_id"]
    candidate_model_path = os.path.join(CANDIDATE_DIR, f"model_{run_id}.joblib")
    candidate_meta_path = os.path.join(CANDIDATE_DIR, f"model_{run_id}.json")

    artifact = {
        "model": model,
        "threshold": meta["threshold"],
        "feature_cols": meta["feature_cols"],
        "model_name": meta["model_name"],
        "run_id": meta["run_id"],
        "run_time": meta["run_time"],
    }
    joblib.dump(artifact, candidate_model_path)

    with open(candidate_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return candidate_model_path, candidate_meta_path


def append_training_history(meta: dict):
    ensure_artifact_dirs()

    row = {
        "run_id": meta["run_id"],
        "run_time": meta["run_time"],
        "model_name": meta["model_name"],
        "threshold": meta["threshold"],
        "primary_metric": meta["primary_metric"],

        "val_accuracy": meta.get("val_accuracy"),
        "val_precision": meta.get("val_precision"),
        "val_recall": meta.get("val_recall"),
        "val_roc_auc": meta.get("val_roc_auc"),
        "val_pr_auc": meta.get("val_pr_auc"),
        "val_f1": meta.get("val_f1"),
        "val_correct_rate": meta.get("val_correct_rate"),
        "val_error_rate": meta.get("val_error_rate"),

        "test_accuracy": meta.get("test_accuracy"),
        "test_precision": meta.get("test_precision"),
        "test_recall": meta.get("test_recall"),
        "test_roc_auc": meta.get("test_roc_auc"),
        "test_pr_auc": meta.get("test_pr_auc"),
        "test_f1": meta.get("test_f1"),
        "test_correct_rate": meta.get("test_correct_rate"),
        "test_error_rate": meta.get("test_error_rate"),

        "test_tn": meta.get("test_tn"),
        "test_fp": meta.get("test_fp"),
        "test_fn": meta.get("test_fn"),
        "test_tp": meta.get("test_tp"),

        "n_features": meta["n_features"],
        "train_rows": meta["train_rows"],
        "val_rows": meta["val_rows"],
        "test_rows": meta["test_rows"],
        "obs_days": meta["obs_days"],
        "pred_days": meta["pred_days"],
        "min_events_in_obs": meta["min_events_in_obs"],
        "threshold_mode": meta["threshold_mode"],
        "target_precision": meta["target_precision"],
    }

    row_df = pd.DataFrame([row])

    if os.path.exists(HISTORY_PATH):
        hist = pd.read_csv(HISTORY_PATH)
        hist = pd.concat([hist, row_df], ignore_index=True)
    else:
        hist = row_df

    hist.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

def load_best_meta():
    if not os.path.exists(BEST_META_PATH):
        return None
    try:
        with open(BEST_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_better_than_best(new_meta: dict, old_meta: dict | None):
    if old_meta is None:
        return True

    metric = new_meta["primary_metric"]
    new_score = new_meta.get(metric)
    old_score = old_meta.get(metric)

    if new_score is None:
        return False
    if old_score is None:
        return True

    try:
        new_score = float(new_score)
        old_score = float(old_score)
    except Exception:
        return False

    if np.isnan(new_score):
        return False
    if np.isnan(old_score):
        return True

    if new_score > old_score + 1e-12:
        return True

    if abs(new_score - old_score) <= 1e-12:
        new_f1 = float(new_meta.get("test_f1", np.nan))
        old_f1 = float(old_meta.get("test_f1", np.nan))
        if np.isnan(old_f1):
            return True
        if np.isnan(new_f1):
            return False
        return new_f1 > old_f1 + 1e-12

    return False


def save_best_artifact(model, meta: dict):
    ensure_artifact_dirs()

    artifact = {
        "model": model,
        "threshold": meta["threshold"],
        "feature_cols": meta["feature_cols"],
        "model_name": meta["model_name"],
        "run_id": meta["run_id"],
        "run_time": meta["run_time"],
    }

    joblib.dump(artifact, BEST_MODEL_PATH)

    with open(BEST_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def train_select_and_save_best(models: dict,
                               X_train,
                               y_train,
                               X_val,
                               y_val,
                               X_test,
                               y_test,
                               feature_cols,
                               train_df,
                               val_df,
                               test_df, progress_callback=None, control_state=None):

    def _is_stopped(state):
        return state.get("stop_flag", False) or state.get("stop", False)

    def _is_paused(state):
        return state.get("pause_flag", False) or state.get("pause", False)

    all_runs = []

    total_models = len(models)

    for idx, (name, model) in enumerate(models.items(), start=1):
        if control_state is not None:
            if _is_stopped(control_state):
                raise RuntimeError("任务已停止")

            while _is_paused(control_state):
                time.sleep(0.2)
                if _is_stopped(control_state):
                    raise RuntimeError("任务已停止")

        if progress_callback is not None:
            progress_callback(
                stage=f"训练模型 {name}",
                current=idx,
                total=total_models
            )

        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]

        thr, (p, r, f1v) = select_threshold(
            y_val,
            val_prob,
            mode=THRESHOLD_MODE,
            target_precision=TARGET_PRECISION
        )

        print(f"\n{name} threshold={thr}")

        val_metrics = evaluate_with_threshold(
            f"{name} VAL",
            y_val,
            val_prob,
            thr
        )

        test_prob = model.predict_proba(X_test)[:, 1]

        test_metrics = evaluate_with_threshold(
            f"{name} TEST",
            y_test,
            test_prob,
            thr
        )

        meta = make_run_meta(
            name,
            thr,
            val_metrics,
            test_metrics,
            feature_cols,
            train_df,
            val_df,
            test_df
        )

        save_candidate_artifact(model, meta)

        append_training_history(meta)

        all_runs.append({
            "model": model,
            "meta": meta
        })

    all_runs = sorted(
        all_runs,
        key=lambda x: x["meta"].get(PRIMARY_METRIC, -np.inf),
        reverse=True
    )

    best_model = all_runs[0]["model"]
    best_meta = all_runs[0]["meta"]

    old_meta = load_best_meta()

    if is_better_than_best(best_meta, old_meta):
        save_best_artifact(best_model, best_meta)
        print("更新最佳模型")
    else:
        print("保留旧最佳模型")

    return best_model, best_meta


# ----------------------------
# 数据加载
# ----------------------------
def load_data(file_obj=None):
    """
    优先读取上传文件；
    如果没有上传文件，则读取仓库 data 目录中的默认文件。
    """
    source_name = None

    if file_obj is not None:
        file_name = file_obj.name.lower()
        source_name = file_obj.name

        if file_name.endswith(".csv"):
            events = pd.read_csv(file_obj)
        elif file_name.endswith(".xlsx"):
            events = pd.read_excel(file_obj)
        else:
            raise ValueError("仅支持 csv 或 xlsx 文件")

    else:
        if not os.path.exists(EVENTS_PATH):
            raise FileNotFoundError(
                f"未上传文件，且默认数据文件不存在：{EVENTS_PATH}"
            )

        source_name = EVENTS_PATH

        if EVENTS_PATH.lower().endswith(".csv"):
            events = pd.read_csv(EVENTS_PATH)
        elif EVENTS_PATH.lower().endswith(".xlsx"):
            events = pd.read_excel(EVENTS_PATH)
        else:
            raise ValueError("默认数据文件仅支持 csv 或 xlsx 文件")

    if "timestamp" not in events.columns:
        raise ValueError("数据中缺少 timestamp 列")

    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")

    print(f"[load_data] 当前读取数据源: {source_name}")
    return events


# ----------------------------
# 特征工程
# ----------------------------
def featurize_user_obs(df):

    g = df.groupby("visitorid")

    feats = pd.DataFrame({
        "visitorid": g.size().index,
        "obs_event_count": g.size().values
    })

    type_counts = pd.crosstab(df["visitorid"], df["event"])

    feats = feats.merge(
        type_counts,
        left_on="visitorid",
        right_index=True,
        how="left"
    ).fillna(0)

    feats.rename(columns={
        "view": "obs_view_count",
        "addtocart": "obs_cart_count",
        "transaction": "obs_trans_count"
    }, inplace=True)

    feats["obs_unique_items"] = g["itemid"].nunique().values

    return feats


# ----------------------------
# 构建监督数据集
# ----------------------------
def build_dataset(events, progress_callback=None, control_state=None):

    def _is_stopped(state):
        return state.get("stop_flag", False) or state.get("stop", False)

    def _is_paused(state):
        return state.get("pause_flag", False) or state.get("pause", False)

    min_time = events["timestamp"].min()
    max_time = events["timestamp"].max()

    anchors = pd.date_range(
        min_time + pd.Timedelta(days=OBS_DAYS),
        max_time - pd.Timedelta(days=PRED_DAYS),
        freq=ANCHOR_FREQ
    )

    rows = []
    total = len(anchors)

    for i, anchor in enumerate(anchors, start=1):

        if control_state is not None:
            if _is_stopped(control_state):
                raise RuntimeError("任务已停止")

            while _is_paused(control_state):
                time.sleep(0.2)
                if _is_stopped(control_state):
                    raise RuntimeError("任务已停止")

        obs_start = anchor - pd.Timedelta(days=OBS_DAYS)
        obs_end = anchor
        pred_end = anchor + pd.Timedelta(days=PRED_DAYS)

        obs = events[
            (events["timestamp"] >= obs_start) &
            (events["timestamp"] < obs_end)
        ]

        pred = events[
            (events["timestamp"] >= obs_end) &
            (events["timestamp"] < pred_end)
        ]

        feats = featurize_user_obs(obs)

        buyers = pred[pred["event"] == "transaction"]["visitorid"].unique()

        feats["label_purchase_next_window"] = feats["visitorid"].isin(buyers).astype(int)
        feats["anchor_time"] = anchor

        rows.append(feats)

        if progress_callback is not None:
            progress_callback(
                stage="构建数据集",
                current=i,
                total=total
            )

    if len(rows) == 0:
        raise ValueError("构建数据集失败：没有生成任何监督样本，请检查 events.csv 时间范围。")

    dataset = pd.concat(rows, ignore_index=True)

    return dataset

# ----------------------------
# 时间切分
# ----------------------------
def time_split_train_val_test(dataset):

    anchor_times = sorted(dataset["anchor_time"].drop_duplicates())
    n_total = len(anchor_times)

    if n_total < 3:
        raise ValueError(
            f"anchor 数量不足，当前只有 {n_total} 个 anchor，至少需要 3 个。"
            f"请增大数据时间范围，或减小 OBS_DAYS / PRED_DAYS，或缩短 ANCHOR_FREQ。"
        )

    # 先切 test，至少保证 test 有 1 个 anchor
    n_train_total = max(2, int(n_total * TRAIN_FRAC))
    if n_train_total >= n_total:
        n_train_total = n_total - 1

    train_val_anchors = anchor_times[:n_train_total]
    test_anchors = anchor_times[n_train_total:]

    # 再在 train_val 内切 val，至少保证 train 和 val 各 1 个 anchor
    n_val = max(1, int(len(train_val_anchors) * VAL_FRAC_WITHIN_TRAIN))
    if n_val >= len(train_val_anchors):
        n_val = 1

    train_anchors = train_val_anchors[:-n_val]
    val_anchors = train_val_anchors[-n_val:]

    if len(train_anchors) < 1 or len(val_anchors) < 1 or len(test_anchors) < 1:
        raise ValueError(
            "时间切分失败：train/val/test 至少有一个为空。"
            f"当前 anchor 总数={n_total}, "
            f"train={len(train_anchors)}, val={len(val_anchors)}, test={len(test_anchors)}。"
            "请增大数据时间范围，或减小 OBS_DAYS / PRED_DAYS，或缩短 ANCHOR_FREQ。"
        )

    train_df = dataset[dataset["anchor_time"].isin(train_anchors)].copy()
    val_df = dataset[dataset["anchor_time"].isin(val_anchors)].copy()
    test_df = dataset[dataset["anchor_time"].isin(test_anchors)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "切分后的数据表为空："
            f"train_df={train_df.shape}, val_df={val_df.shape}, test_df={test_df.shape}。"
            "请检查时间窗口设置和数据时间范围。"
        )

    return train_df, val_df, test_df

def run_pipeline(file_obj=None, progress_callback=None, control_state=None):
    ensure_artifact_dirs()

    events = load_data(file_obj=file_obj)
    dataset = build_dataset(events, progress_callback=progress_callback, control_state=control_state)

    bad_cols = ["visitorid", "anchor_time", "label_purchase_next_window"]
    feature_cols = [c for c in dataset.columns if c not in bad_cols]

    dataset = sanitize_features(dataset, feature_cols)
    train_df, val_df, test_df = time_split_train_val_test(dataset)

    X_train = train_df[feature_cols].values
    y_train = train_df["label_purchase_next_window"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label_purchase_next_window"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label_purchase_next_window"].values

    models = build_models(random_state=42)

    best_model, best_meta = train_select_and_save_best(
        models,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols,
        train_df,
        val_df,
        test_df,
        progress_callback=progress_callback,
        control_state=control_state
    )

    cv_metrics = []
    if ENABLE_ROLLING_CV:
        cv_metrics = rolling_time_cv(
            dataset,
            build_models(random_state=42)["HistGB"],
            feature_cols,
            n_folds=CV_N_FOLDS,
            min_train_anchors=CV_MIN_TRAIN_ANCHORS,
            progress_callback=progress_callback,
            control_state=control_state
        )

    if progress_callback is not None:
        progress_callback(stage="计算特征重要性", current=1, total=1)

    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="average_precision"
    )

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    dataset.to_csv("supervised_dataset_user_level.csv", index=False)
    imp.to_csv("feature_importance_perm.csv", index=False)

    pd.DataFrame([
        {k: v for k, v in best_meta.items() if k != "feature_cols"}
    ]).to_csv(
        LAST_RUN_META_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    if cv_metrics:
        pd.DataFrame(cv_metrics).to_csv(
            ROLLING_CV_PATH,
            index=False,
            encoding="utf-8-sig"
        )

    visual_metrics = {
        "accuracy": best_meta.get("test_accuracy"),
        "precision": best_meta.get("test_precision"),
        "recall": best_meta.get("test_recall"),
        "roc_auc": best_meta.get("test_roc_auc"),
        "pr_auc": best_meta.get("test_pr_auc"),
        "f1": best_meta.get("test_f1"),
        "correct_rate": best_meta.get("test_correct_rate"),
        "error_rate": best_meta.get("test_error_rate"),
        "tn": best_meta.get("test_tn"),
        "fp": best_meta.get("test_fp"),
        "fn": best_meta.get("test_fn"),
        "tp": best_meta.get("test_tp")
    }

    return {
        "dataset": dataset,
        "best_meta": best_meta,
        "importance": imp,
        "cv_metrics": cv_metrics,
        "visual_metrics": visual_metrics
    }
# ----------------------------
# Rolling Time CV
# ----------------------------
def rolling_time_cv(dataset, model, feature_cols, n_folds=3, min_train_anchors=3, progress_callback=None, control_state=None):
    def _is_stopped(state):
        return state.get("stop_flag", False) or state.get("stop", False)

    def _is_paused(state):
        return state.get("pause_flag", False) or state.get("pause", False)

    anchor_times = sorted(dataset["anchor_time"].drop_duplicates())
    total_anchors = len(anchor_times)

    if total_anchors < (min_train_anchors + n_folds):
        print("[RollingCV] anchor 数量不足，跳过 rolling CV")
        return []

    fold_metrics = []

    for i in range(n_folds):

        if control_state is not None:
            if _is_stopped(control_state):
                raise RuntimeError("任务已停止")

            while _is_paused(control_state):
                time.sleep(0.2)
                if _is_stopped(control_state):
                    raise RuntimeError("任务已停止")

        train_end_idx = min_train_anchors + i
        val_idx = train_end_idx

        if progress_callback is not None:
            progress_callback(
                stage="Rolling CV",
                current=i + 1,
                total=n_folds
            )

        if val_idx >= total_anchors:
            break

        train_anchors = anchor_times[:train_end_idx]
        val_anchor = anchor_times[val_idx]

        train_df = dataset[dataset["anchor_time"].isin(train_anchors)].copy()
        val_df = dataset[dataset["anchor_time"] == val_anchor].copy()

        if train_df.empty or val_df.empty:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df["label_purchase_next_window"].values

        X_val = val_df[feature_cols].values
        y_val = val_df["label_purchase_next_window"].values

        m = build_models(random_state=42)["HistGB"]
        m.fit(X_train, y_train)

        val_prob = m.predict_proba(X_val)[:, 1]
        thr, _ = select_threshold(
            y_val,
            val_prob,
            mode=THRESHOLD_MODE,
            target_precision=TARGET_PRECISION
        )

        metrics = evaluate_with_threshold(
            f"RollingCV Fold {i+1}",
            y_val,
            val_prob,
            thr
        )

        metrics["fold"] = i + 1
        metrics["val_anchor"] = val_anchor
        fold_metrics.append(metrics)

    return fold_metrics


# ----------------------------
# 主流程
# ----------------------------
def main():
    if not (
            os.path.exists(EVENTS_PATH) and
            os.path.exists(CATEGORY_TREE_PATH) and
            os.path.exists(ITEM_PROPS_PATH)
    ):
        raise FileNotFoundError(
            f"数据文件路径不存在，请检查 data 目录下是否存在："
            f"{os.path.basename(EVENTS_PATH)} / "
            f"{os.path.basename(CATEGORY_TREE_PATH)} / "
            f"{os.path.basename(ITEM_PROPS_PATH)}"
        )

    ensure_artifact_dirs()

    # 1) 读取数据
    events = load_data()

    # 2) 构建监督数据集
    dataset = build_dataset(events)

    # 3) 特征列
    bad_cols = ["visitorid", "anchor_time", "label_purchase_next_window"]
    feature_cols = [c for c in dataset.columns if c not in bad_cols]

    # 4) 清洗特征
    dataset = sanitize_features(dataset, feature_cols)
    print("[OK] Features sanitized.")

    # 5) 时间切分
    train_df, val_df, test_df = time_split_train_val_test(dataset)

    print("dataset shape:", dataset.shape)
    print("anchor count:", dataset["anchor_time"].nunique())
    print("train size:", train_df.shape)
    print("val size:", val_df.shape)
    print("test size:", test_df.shape)

    X_train = train_df[feature_cols].values
    y_train = train_df["label_purchase_next_window"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label_purchase_next_window"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label_purchase_next_window"].values

    # 6) 训练多个模型并自动保存最佳模型
    models = build_models(random_state=42)

    best_model, best_meta = train_select_and_save_best(
        models,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols,
        train_df,
        val_df,
        test_df
    )

    # 7) Rolling CV
    if ENABLE_ROLLING_CV:
        print("\n================ Rolling Time CV ================")
        cv_model = build_models(random_state=42)["HistGB"]
        cv_metrics = rolling_time_cv(
            dataset,
            cv_model,
            feature_cols,
            n_folds=CV_N_FOLDS,
            min_train_anchors=CV_MIN_TRAIN_ANCHORS
        )

        if cv_metrics:
            cv_df = pd.DataFrame(cv_metrics)
            print(cv_df.to_string(index=False))
            cv_df.to_csv(ROLLING_CV_PATH, index=False, encoding="utf-8-sig")
        else:
            print("No CV metrics produced.")
        print("=================================================\n")

    # 8) 对本次最佳模型做 permutation importance
    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="average_precision"
    )

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    print(f"\nTop-30 permutation importance for [{best_meta['model_name']}]:")
    print(imp.head(30).to_string(index=False))

    # 9) 保留你原来的输出
    dataset.to_csv("supervised_dataset_user_level.csv", index=False)
    imp.to_csv("feature_importance_perm.csv", index=False)

    # 10) 保存本次最佳 meta 快照
    pd.DataFrame([
        {k: v for k, v in best_meta.items() if k != "feature_cols"}
    ]).to_csv(
        LAST_RUN_META_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved:")
    print("  - supervised_dataset_user_level.csv")
    print("  - feature_importance_perm.csv")
    print(f"  - {BEST_MODEL_PATH}")
    print(f"  - {BEST_META_PATH}")
    print(f"  - {HISTORY_PATH}")
    print(f"  - {LAST_RUN_META_PATH}")
    if ENABLE_ROLLING_CV:
        print(f"  - {ROLLING_CV_PATH}")


if __name__ == "__main__":
    main()
