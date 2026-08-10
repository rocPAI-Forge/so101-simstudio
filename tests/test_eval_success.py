from simstudio.common.eval_success import check_pick_success


def test_pick_success_inside_container():
    assert check_pick_success((0.3, 0.2, 0.015))


def test_pick_success_outside_xy():
    assert not check_pick_success((0.1, 0.2, 0.015))


def test_pick_success_too_high():
    assert not check_pick_success((0.3, 0.2, 0.05))


def test_pick_success_none():
    assert not check_pick_success(None)
