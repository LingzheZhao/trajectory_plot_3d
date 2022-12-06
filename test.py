import matplotlib.pyplot as plt
from evo.tools import file_interface
from evo.tools import plot
from evo.core.sync import associate_trajectories
import camera_plot


def test_compare_two_trajactories(traj_01_, traj_gt_):
    a = plot.PlotCollection("test")
    fig = plt.figure()
    plot_mode = plot.PlotMode["xyz"]
    ax = plot.prepare_axis(fig, plot_mode)

    plot.traj(ax, plot_mode, traj_01_,
              style="-",
              color="r",
              label="method 01",
              alpha=0.5)


    plot.traj(ax, plot_mode, traj_gt_,
              style="--",
              color="gray",
              label="GT",
              alpha=0.5)

    # marker_scale is the size of the pose pyramids
    camera_plot.compare_two_trajactories(ax, traj_gt_, traj_01_, marker_scale=0.005)

    # This is an alternative to the pyramids
    # plot.draw_coordinate_axes(ax, traj_01_, plot_mode, 0.02)
    # plot.draw_coordinate_axes(ax, traj_gt_, plot_mode, 0.02)

    a.add_figure("compare two trajactories", fig)
    a.show()


def test_compare_three_trajactories(traj_01_, traj_02_, traj_gt_):
    a = plot.PlotCollection("test")
    fig = plt.figure()
    plot_mode = plot.PlotMode["xyz"]
    ax = plot.prepare_axis(fig, plot_mode)

    plot.traj(ax, plot_mode, traj_01_,
              style="-",
              color="r",
              label="method 01",
              alpha=0.5)

    plot.traj(ax, plot_mode, traj_02_,
              style="-",
              color="b",
              label="method 2",
              alpha=0.5)

    plot.traj(ax, plot_mode, traj_gt_,
              style="--",
              color="gray",
              label="GT",
              alpha=0.5)

    # This is an alternative to the pyramids
    # plot.draw_coordinate_axes(ax, traj_01_, plot_mode, 0.02)
    # plot.draw_coordinate_axes(ax, traj_02_, plot_mode, 0.02)
    # plot.draw_coordinate_axes(ax, traj_gt_, plot_mode, 0.02)

    # marker_scale is the size of the pose pyramids
    camera_plot.draw_pyramids(ax, traj_01_, marker_scale=0.005, color='r')
    camera_plot.draw_pyramids(ax, traj_02_, marker_scale=0.005, color='b')
    camera_plot.draw_pyramids(ax, traj_gt_, marker_scale=0.005, color='gray')
    a.add_figure("compare three trajactories", fig)
    a.show()


def main():
    FILE_GT = "./test/data/freiburg1_xyz-groundtruth.txt"
    FILE_01 = "./test/data/freiburg1_xyz-rgbdslam.txt"
    FILE_02 = "./test/data/freiburg1_xyz-ORB_kf_mono.txt"

    traj_gt = file_interface.read_tum_trajectory_file(FILE_GT)
    traj_01 = file_interface.read_tum_trajectory_file(FILE_01)
    traj_02 = file_interface.read_tum_trajectory_file(FILE_02)

    gt_synced01, traj_01_synced = associate_trajectories(
        traj_gt, traj_01, max_diff=0.1)
    gt_synced02, traj_02_synced = associate_trajectories(
        traj_gt, traj_02, max_diff=0.1)

    traj_01_synced.align(gt_synced01, correct_scale=True)
    traj_02_synced.align(gt_synced02, correct_scale=True)
    test_compare_two_trajactories(traj_01_synced, gt_synced01)
    test_compare_three_trajactories(traj_01_synced, traj_02_synced, traj_gt)


if __name__ == "__main__":
    main()
