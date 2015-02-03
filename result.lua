require('torch')
require('nn')
require('xlua')

-- Download and load test data
print('==> downloading dataset')
www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/housenumbers/'
test_file = 'test_32x32.t7'
if not paths.filep(test_file) then
   os.execute('wget ' .. www .. test_file)
end

print('==> loading data')
loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = 26032
}

-- Load model
print('==> loading model')
model_file = 'model.net'
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
