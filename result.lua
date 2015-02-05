require('torch')
require('nn')
require('xlua')

-- Download and load data
print('==> downloading dataset')
www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/housenumbers/'
train_file = 'train_32x32.t7'
test_file = 'test_32x32.t7'
if not paths.filep(train_file) then
   os.execute('wget ' .. www .. train_file)
end
if not paths.filep(test_file) then
   os.execute('wget ' .. www .. test_file)
end

print('==> loading data')
loaded = torch.load(train_file,'ascii')
trainData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = 26032
}

loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = 26032
}

print '==> normalizing data'
trainData.data = trainData.data:float()
testData.data = testData.data:float()

channels = {'r','g','b'}
-- Calculating mean, std of training data
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Load model
print('==> loading model')
model_file = 'model_sr_14.net'
if not paths.filep(model_file) then
   os.exit('Model file name needs to be model.net')
end
model = torch.load(model_file)

-- Testing on test data
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
   preds[t] = label[1]
end

-- Output to a csv file
print('==> saving predictions')
file = io.open('predictions.csv', 'w')
file:write('Id,Prediction\n')
for i, p in ipairs(preds) do
   file:write(i..','..p..'\n')
end
file:close()
