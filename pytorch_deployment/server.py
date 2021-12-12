from settings import *
from torch.utils.data import Dataset
import torch
import torch.utils.data as data_utils
import pandas as pd
from flask import Flask, render_template



## test data
class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = pd.read_csv(data_dir)
test_data = test_data.drop('SAGO_YN', axis = 1)

test_data_sample = test_data.sample(10)
print(test_data_sample.shape)

test_data = testData(torch.FloatTensor(test_data_sample.values))
test_loader = data_utils.DataLoader(test_data, batch_size=1, shuffle=False)

checkpoint = torch.load(model_path, map_location = 'cpu')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(model.eval())



app = Flask(__name__)


# @app.route('/', methods = ['GET', 'POST'])
@app.route('/predict', methods = ['GET'])

def predict():
    with torch.no_grad():

        y_pred_result = pd.DataFrame(columns = ['교통사고_분류예측율', '교통사고_추론결과'])

        for X_batch in test_loader:
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)

            y_pred_prob = torch.FloatTensor.item(y_test_pred)
            y_pred_prob = round(y_pred_prob, 2)

            if y_pred_prob >= 0.5:
                y_pred_label = labels[1]
            else:
                y_pred_label = labels[0]

            new_row = {'교통사고_분류예측율': y_pred_prob, '교통사고_추론결과': y_pred_label}
            y_pred_result = y_pred_result.append(new_row, ignore_index=True)

    return y_pred_result


pd_result_table = predict()
print(pd_result_table.head(7))


@app.route('/html_table/', methods = ['GET'])
# @app.route('/', methods = ['GET'])

def html_table():
    pd_predict_table = predict()
    # print(pd_predict_table.shape)

    result_html = pd_predict_table.to_html(header='True', table_id='table')

    ''''''
    result_output = open('templates/result_output.html', 'w')
    result_output.write(result_html)
    ''''''

    return result_html



@app.route('/view_table', methods = ['GET'])
def view_table():

   return render_template('result_output.html')



if __name__ == '__main__':
   app.run(host = 'localhost', port = PORT)