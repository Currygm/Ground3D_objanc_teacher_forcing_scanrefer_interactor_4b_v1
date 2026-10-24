"""Stage-relative checkpoint cadence for a fresh two-stage run."""

def stage_progress(completed_epochs, answer_epochs):
    if completed_epochs < 1 or answer_epochs < 1:
        raise ValueError("Epoch counts must be positive")
    if completed_epochs <= answer_epochs:
        return "answer_only", completed_epochs
    return "joint", completed_epochs-answer_epochs


def should_save_epoch(completed_epochs, answer_epochs, total_epochs, interval):
    if interval < 1 or not 0 < answer_epochs < total_epochs:
        raise ValueError("Invalid two-stage checkpoint schedule")
    if not 0 < completed_epochs <= total_epochs:
        return False
    phase, stage_epoch = stage_progress(completed_epochs, answer_epochs)
    if phase == "answer_only":
        return completed_epochs == answer_epochs
    return stage_epoch % interval == 0 or completed_epochs == total_epochs
