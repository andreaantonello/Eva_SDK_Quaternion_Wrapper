# SDK tool for TCP rotation (aeronautical convention) 
This tool allows to rotate the robot around the tool center point (TCP). The transform is 
defined as successive yaw, pitch, roll rotations around the Z, Y and X axes, respectively.
This tool will return the joint angles corresponding to the desired TCP transform.

# Setup
First of all, the TCP configuration must be set up. 
The parameters can be found in the file `config/use_case_config.yaml`.
In this file, the connection params to the robot will have to be inserted (IP and token).

# Usage

* First of all, it is mandatory to bring the robot in the *flat* position. This is the 
**zeroed TCP position** (yaw=0, pitch=0, roll=0), and will be used as the initial
condition over which all the rotations will be applied. Once the configuration has been 
obtained (in the form of a 1x6 joint vector), the values in radians must be inserted in the
`config/use_case_config.yaml` file, under the YAML entry `['EVA']['tcp']['initial_pos']`.

* At this point, you can run the main file `run_this.py`. You will be asked to verify the 
**zeroed TCP position** you've previously entered. Input `yes`:


```
    Have you already verified the TCP position? [yes/no]
    $ yes
```

* You will be then shown with a 3D model of Eva, in the **zeroed TCP position**. The TCP will be
pictured as a simplified set of lines, with a dummy payload at the end (credit card sized). A frame
of reference will shown as well. 

NOTE: an internal tool takes care of straightening the head to 
the closest axis (i.e. if the orientation is slightly off axis, for example 85 deg, the code will
automatically set this to 90 deg. Make sure this corrected position is as expected). Ultimately, 
this will modify the **zeroed TCP position** into the **corrected zeroed TCP position**


```
    WARNING: Orientation has been straighted to the closest, adjacent axis!
        -> Original quaternion was: [-0.0042036995, -3.885815e-16, 0.9999912, 4.3711e-08] 
        -> Straightened quaternion is now: [-0.0, -0.0, 1.0, 0.0] 
```

* If the **corrected zeroed TCP position** is as expected, please proceed by answering yes to the prompted
question:


```
    Is the initial pose and TCP configuration correct?
    $ yes
```    
    
    
* At this point, you have successfully set-up the use case, and the transformation matrix between your TCP
and the end-effector plate (`tcp_transform` in the script) will be stored internally for all the subsequent
manipulation, along with the **corrected zeroed TCP position** (`q_corrected` in the script). Specifically, this is the function
that will return the above-mentioned outputs:

```
    q_corrected, tcp_transform = setup.run()
```
    
* Finally, the transformation tool around the TCP can be used. The function to obtain this is the following:

```
    q_rotated = transform_tcp_to_joint_angles(q_corrected, tcp_transform, yaw=0, pitch=0, roll=0)
```

* The inputs are the setup values `q_corrected`, `tcp_transform` and the Euler angles `yaw`,
`pitch`, `roll`. The code uses the ZYX Euler angles order (known also as the aeronautical yaw, pitch and roll).
Be aware that gimbal lock will around if the `pitch` is set to pi/2. The function returns the corresponding
joints angle 1x6 vector.  


# Visual tool

It is possible to use the visualizer anywhere in the code. It is important to set up the figure environment 
as follows:

```
    # Instantiate plotting class
    plotEva = PlotEva(eva, eva_model)
    
    # Load plotter
    plot = pv.Plotter()
    
    # Plot frame of reference
    plotEva.plot_frame(plot, frame_to_be_plotted)
   
    # Plot joint position
    plot = plotEva.plot_pose(plot, position_to_be_plotted, tcp=True, frames=False)
    
    # Show plot window
    plot.show()
```


That is, the plotting class `PlotEva(eva, eva_model)` will have to be instantiated, and the plotter environment 
set up through `plot = pv.Plotter()`. The plotting functions are:

* `plot_pose(plot, position_to_be_plotted, tcp=True, frames=False)`: this is for the full robot
plot. It takes as input the `plot` environment, the 1x6 joint positions (`position_to_be_plotted`). Two optional
inputs are for displaying the tcp and the frames of reference. 
* `plot_frame(plot, frame_to_be_plotted)`: this is for plotting a frame of reference. It takes as inputs 
 the `plot` environment and the frame of reference, expressed as a 4x4 
 transform matrix (`frame_to_be_plotted`)
 * `plot_sphere(plot, pos, color_user='black')`: this is for plotting a sphere. It takes as inputs 
 the `plot` environment and the center-point position of the sphere (`pos`). An optional
input is for the sphere color 