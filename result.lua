require('torch')
require('nn')
require('xlua')

-- Download and load training, test data
print('==> downloading dataset')
www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/housenumbers/'
train_file = 'train_32x32.t7'
test_file = 'test_32x32.t7'
extra_file = 'extra_32x32.t7'

if not paths.filep(train_file) then
   os.execute('wget ' .. www .. train_file)
   end
if not paths.filep(test_file) then
   os.execute('wget ' .. www .. test_file)
end
if not paths.filep(extra_file) then
   os.execute('wget ' .. www .. extra_file)
end

-- For model 2, we used extra data, thus training size is 73257 + 531131
trsize = 73257 + 531131
tesize = 26032

-- Load training
print('==> loading data')
loaded = torch.load(train_file,'ascii')
trainData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return trsize end
}

-- Load extra
loaded = torch.load(extra_file,'ascii')
trdata = torch.Tensor(trsize,3,32,32)

-- Combine full and extra
trdata[{ {1,(#trainData.data)[1]} }] = trainData.data
trdata[{ {(#trainData.data)[1]+1,-1} }] = loaded.X:transpose(3,4)
trlabels = torch.Tensor(trsize)
trlabels[{ {1,(#trainData.labels)[1]} }] = trainData.labels
trlabels[{ {(#trainData.labels)[1]+1,-1} }] = loaded.y[1]
trainData = {
   data = trdata,
   labels = trlabels,
   size = function() return trsize end
}

-- Load test data
loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = 26032
}

trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- Preprocess data in the same way as described in 1_data.lua
print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module,
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

-- Load trained model
print('==> loading model')
model_file = 'results/model.net'
if not paths.filep(model_file) then
   os.exit('please check model file name: {}'.format(model_file))
end
model = torch.load(model_file)

-- Test on test data
print('==> making predictions')
model:evaluate()
preds = {}
for t = 1,testData.size do
   -- show progress
   xlua.progress(t, testData.size)
   
   -- get new sample
   local input = testData.data[t]
   input = input:double()

   -- predict
   local pred = model:forward(input)
   local label = torch.LongTensor()
   local _max = torch.FloatTensor()
   _max:max(label, pred:float(), 1)
   preds[t] = label
end

-- Output to a csv file
print('==> saving predictions')
file = io.open('predictions.csv', 'w')
file:write('Id,Prediction\n')
for i, p in ipairs(preds) do
   file:write(i..','..p..'\n')
end
file:close()
