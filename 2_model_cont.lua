----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- models:
--   + linear
--   + 2-layer neural network (MLP)
--   + convolutional network (ConvNet)
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 2_model.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to play with the model.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
model_file = 'results/model.net'
model = torch.load(model_file)
----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
---- Visualization is quite easy, using gfx.image().
--
--if opt.visualize then
--   if opt.model == 'convnet' then
--      print '==> visualizing ConvNet filters'
--      gfx.image(model:get(1).weight, {zoom=2, legend='L1'})
--      gfx.image(model:get(5).weight, {zoom=2, legend='L2'})
--   end
--end
