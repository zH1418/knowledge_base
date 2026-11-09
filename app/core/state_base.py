import copy

_BASE_STATE: dict = {
    "task_id": "",
}


def create_base_state(**overrides) -> dict:
    state = copy.deepcopy(_BASE_STATE)
    state.update(overrides)
    return state


def get_base_state() -> dict:
    return copy.deepcopy(_BASE_STATE)
