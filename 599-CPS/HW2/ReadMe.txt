A brief explanation of simulink model:
----------------------------------------
For adaptive cruise control, there's a speed control mode and a spacing control mode.

The speed control mode is where the host car tries to maintain the reference speed set
by the driver. The spacing control mode is where the lead car is too near and host car
tries to maintain a safe distance from the lead car. The control algorithm needs to make
a decision based on the sensor inputs when to switch between the speed control vs spacing
control modes.

In my control algorithm, the controller is pretty straight forward. It tries to set
acceleration based on a PID controller from an error with respect to a desired speed
in both the above scenarios. In speed control mode, the error is the difference between
driver set reference velocity and the host car's current velocity.
In the spacing control mode, the host car should match the velocity of lead car
unless the lead car's speed is more than driver set reference velocity. (Though it's
somewhat arguable whethere spacing control needs to be triggered when lead car's velocity
is more than reference speed). For this reason, in spacing control mode, I used
the error as the minimum of relative velocity w.r.t lead car (this is calculated as 
derivative of distance between the cars) and difference between host car velocity and reference speed.

Now when should we switch between speed control and spacing control? Ideally, the spacing
control must be triggered at a distance such that, by the time host car matches lead car's
velocity, the distance between the two cars will be about and less than dSafe distance.
I chose this distance from an experimental setup - I used the provided car models and
used the minimum possible acceleration for the lead car (-1) and maximum possible velocity
of host car (reference velocity) and some velocity of lead car in between 0 and reference velocity
(as the starting point for lead car) and used the same PID (indeed only P) controller 
that I was using in my simulink model to find out the relative distance covered by host car
w.r.t. lead car. And It seemed to me that this distance has a relationship with the reference
velocity and the proportional gain of PID controller , hence I used that as the reference
distance and added dSafe to it (so that the host car will match the velocity before dSafe distance).

Since I chose the worst possible values for cacluating the distance, the above algorithm
is very conservative, so in the cases where lead car is at a constant velocity less than
reference velocity, the host car starts spacing control sooner than necessary so it ends
up maintaining more distance than dSafe distance. I was trying to come up with a more
precise mathematical model for figuring this out but could not yet come up with something
that works reliably in all cases. Since mantaining dSafe was the strict and safe requirement,
I gave priority to this requirement and hence used the more conservative distance to trigger
this switch. (I tried different models earlier using the relative velocities instead of distance
but I thought this is the model which works reasonably well to meet different requirements).

I tried higher values for proportional gains and even other I and D gain terms in PID control
but it seems to me that the set P gain value (0.7) results in a stable acceleration that follows
the requirements smoothly. (I don't know if there was some mistake in my setup - when I keep the
proportional gain term and tried 1 as differential gain term, simulation itself fails and I did
not have a chance to investigate why. Since the simulations themselves fail, I am unable
to see the graphs of different values to understand what's going wrong.)