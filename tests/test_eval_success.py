from simstudio.common.eval_success import check_pick_success


def test_pick_success_inside_container():
    assert check_pick_success((0.07, 0.065, 0.015))


def test_pick_success_outside_xy():
    assert not check_pick_success((0.1, 0.2, 0.015))


def test_pick_success_too_high():
    assert not check_pick_success((0.07, 0.065, 0.05))


def test_pick_success_none():
    assert not check_pick_success(None)


def test_cube_over_container_xy_only():
    from simstudio.common.eval_success import cube_over_container

    assert cube_over_container((0.07, 0.065, 0.12))
    assert not cube_over_container((0.27, 0.20, 0.015))
    assert not cube_over_container(None)


def test_gripper_near_cube_xy_and_z():
    from simstudio.common.eval_success import gripper_near_cube

    cube = (0.27, 0.20, 0.013)
    assert gripper_near_cube((0.27, 0.20, 0.04), cube)
    assert gripper_near_cube((0.27, 0.20, 0.09), cube, xy_m=0.06, z_min_m=0.02, z_max_m=0.14)
    assert not gripper_near_cube((0.209, -0.028, 0.15), cube)
    assert not gripper_near_cube((0.27, 0.20, 0.20), cube)
    assert not gripper_near_cube((0.27, 0.20, 0.09), None)
