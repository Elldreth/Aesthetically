import threading
import time

from app.jobs import JobManager


def _wait(job, timeout=5):
    t0 = time.time()
    while job.state not in ("done", "failed", "cancelled") and time.time() - t0 < timeout:
        time.sleep(0.01)
    return job.state


def test_job_runs_and_returns_result():
    m = JobManager()
    def fn(progress, cancel):
        progress["total"] = 3
        for i in range(3):
            progress["done"] = i + 1
        return {"ok": True}

    job = m.submit("test", "x", fn)
    assert _wait(job) == "done"
    assert job.result == {"ok": True}
    assert job.progress["done"] == 3


def test_jobs_serialize_in_order():
    m = JobManager()
    order = []

    def fn1(progress, cancel):
        order.append("1-start"); time.sleep(0.1); order.append("1-end")

    def fn2(progress, cancel):
        order.append("2-start"); order.append("2-end")

    j1 = m.submit("test", "1", fn1)
    j2 = m.submit("test", "2", fn2)
    _wait(j1); _wait(j2)
    assert order == ["1-start", "1-end", "2-start", "2-end"]


def test_cancel_queued_job_does_not_run():
    m = JobManager()
    block = threading.Event()
    ran = {"v": False}

    def slow(progress, cancel):
        block.wait(2)

    def never(progress, cancel):
        ran["v"] = True

    j1 = m.submit("test", "slow", slow)
    j2 = m.submit("test", "never", never)
    # j1 holds the single worker; cancel j2 while it is still queued
    time.sleep(0.05)
    assert m.cancel(j2.id) is True
    block.set()
    _wait(j1); _wait(j2)
    assert j2.state == "cancelled"
    assert ran["v"] is False


def test_failed_job_records_error():
    m = JobManager()
    def boom(progress, cancel):
        raise ValueError("nope")

    job = m.submit("test", "b", boom)
    assert _wait(job) == "failed"
    assert "nope" in (job.error or "")


def test_active_count_and_latest():
    m = JobManager()
    block = threading.Event()
    j = m.submit("scan", "folderA", lambda p, c: block.wait(2))
    time.sleep(0.05)
    assert m.active_count() == 1
    assert m.latest("scan").id == j.id
    block.set()
    _wait(j)
    assert m.active_count() == 0
