----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining some tools')

-- model:
local t = require 'model'
local model = t.model
local loss = t.loss

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes) -- faces: yes, no

-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Batch test:
local inputs = torch.Tensor(yuvTestData:size(),yuvTestData.data:size(2),
         yuvTestData.data:size(3), yuvTestData.data:size(4)) -- get size from data
local targets = torch.Tensor(opt.batchSize)
if opt.type == 'cuda' then
   inputs = inputs:cuda()
   targets = targets:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

-- test function
function test(testData)
   -- local vars
   local time = sys.clock()

print(sys.COLORS.red .. '==> ADSFSASFSDFSF LOADING MODEL:')
-- network = torch.load('model.net')
-- network_fov = 32
-- network_sub = 4
-- softnorm = network.modules[1]
-- hardnet = nn.Sequential()
-- for i = 2,#network.modules do
--    hardnet:add(network.modules[i])
-- end
model = torch.load('face.net')



   -- test over test data
   print(sys.COLORS.red .. '==> testing on test set:')
   print(sys.COLORS.red .. '==> size:' .. testData:size())
   for t = 1,testData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, testData:size())

      -- batch fits?
      -- if (t + opt.batchSize - 1) > testData:size() then
      --   break
      -- end

      -- create mini batch
      local idx = 1
      for i = t,testData:size()  do
         print(i)
         inputs[idx] = testData.data[i]
         targets[idx] = testData.labels[i]
         idx = idx + 1
      end

      -- test sample
      local preds = model:forward(inputs)

      print('predicates ', preds)

      -- confusion
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end
   confusion:zero()

end

-- Export:
return test
