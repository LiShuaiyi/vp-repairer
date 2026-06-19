import open3d as o3d
import numpy as np

def modify_pcd_coordinates(input_file, output_file, offset):
    pcd = o3d.io.read_point_cloud(input_file)
    
    points = np.asarray(pcd.points)
    
    points += offset
    
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(output_file, pcd)

if __name__ == "__main__":
    input_file = "/home/han/Documents/TUM/Traj_Repair/commonroad-repairer.worktrees/repair-autoware/autoware-repair-docker/map/DEU_GarchingCampus-2_stop_line_6/pointcloud_map.pcd"
    output_file = "/home/han/Documents/TUM/Traj_Repair/commonroad-repairer.worktrees/repair-autoware/autoware-repair-docker/map/DEU_GarchingCampus-2_stop_line_6/pointcloud_map.pcd"
    offset = np.array([0, 0, 1])
    
    modify_pcd_coordinates(input_file, output_file, offset)
