"""单条样本级重试与超时（error / 超时均重试，间隔 1s，最多 5 次重试 = 6 次尝试）."""
import concurrent.futures
import os
import time
from typing import Callable, Tuple, TypeVar

T = TypeVar("T")

# 失败后重试次数（不含首次）；MARC_SAMPLE_MAX_RETRIES 可覆盖
SAMPLE_MAX_RETRIES = int(os.environ.get("MARC_SAMPLE_MAX_RETRIES", "5"))
SAMPLE_RETRY_INTERVAL_SEC = float(os.environ.get("MARC_SAMPLE_RETRY_INTERVAL_SEC", "1.0"))
SAMPLE_TIMEOUT_SEC = float(os.environ.get("MARC_SAMPLE_TIMEOUT_SEC", "300.0"))  # 5 min per sample


class SampleTimeoutError(TimeoutError):
    """单条样本 wall-clock 超过 SAMPLE_TIMEOUT_SEC."""


def call_sample_with_retry(
    fn: Callable[[], T],
    sample_idx: int,
    log_fn=None,
    max_retries: int = SAMPLE_MAX_RETRIES,
    interval_sec: float = SAMPLE_RETRY_INTERVAL_SEC,
    timeout_sec: float = SAMPLE_TIMEOUT_SEC,
) -> Tuple[T, int]:
    """执行 fn；失败或超时则间隔重试，成功返回 (result, attempts_used_after_first)."""
    last_err = None
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(fn)
                try:
                    result = fut.result(timeout=timeout_sec)
                    return result, attempt
                except concurrent.futures.TimeoutError:
                    raise SampleTimeoutError(
                        f"sample {sample_idx + 1} exceeded {timeout_sec:.0f}s wall-clock"
                    )
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                if log_fn:
                    log_fn(
                        f"  [RETRY] sample {sample_idx + 1} attempt {attempt + 1}/{total_attempts} "
                        f"failed: {type(e).__name__}: {e}, retry in {interval_sec}s..."
                    )
                time.sleep(interval_sec)
    raise last_err
