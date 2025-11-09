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






Nov 19 3 -4 
BE FREE MAN

Data science related topic next wednesday and next sem something like that

conductivity: empirical law
circshift: the average of the cells around it

understand the upside down triangle


V(x) = k(v?)*(dP(x))/ dx
k(upside down triangle)P = v
P2 - p1 = kv
deltaP/deltax = kv
dp/dt + UpsideTri * V = 0
k(x)= k1 + k2 sin(2xnx/L) 
conductivity matrix(x,y)
blundry conditions on 2d is on inlet outlet, rest is decided by conductivity, 

given changes in system how would pressure evolve with time
k = 1 - alpha?t







Work on the ml code try to understand it, check in with kudulis code too 



add some noise

literatue search for future ideas



make the points not clump, spead similarly


different distributions for disk

keep 400 samples, reduce the steps, maybe to like 100

mess with differnet binaries, ...
cross correlation to see what things give better results, sparsed samples... 



try giving all information and sensor points, 
cross correlation, score how well it predicts, plot scatter to avoid fluke from random cherry picks
for all images in testing set, do compare, trends mean... 
Systematic proof
40 - 80 more than 150 channel just gets wider

replusion between points, some points blackout (cant drill in a mountain example)

30-80 changes a lot find where change occurs