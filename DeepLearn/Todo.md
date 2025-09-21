gradient boosting
ramp rate: branch more
stickyness
Path prediction, also parameter prediction
binary river, uniform
differnt stickyness levels for sediment
pressure at certain points
pinn darcy's law
Binary river for now
test function with different pipes, one is long tube same height, other is tube of varying heights, wide, short
Pinn darcy




09/19
Change simple equaton to fit to our problem
They used complete form for plotting
start with smaller example\
Permearbility
perocity
predicting perocity
Inverse beam problem, can you solve for the force distribution
for inverse: learn the pressure or heading see if thats helpful

local velocity
If were given pressure at entrance, exit
If were predicting pressure, input would be 
We have initial pressure. We can learn pressure at all points given only a few samples
We can have pressure and conductivity and predict pressure everywhere and have it follow ∇· (𝜅∇𝑃) = 0

first loss can be image type loss mse, 
physics loss is 

we dont have permeability we have conductivity

We need boundry conditions



I need to give code, understand the variables within the simulation and see what can be done with them.

Find an example that solves the inverse problem
Inverse version of beam
take displacement, try to return pressure