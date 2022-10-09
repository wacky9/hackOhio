import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#import PySimpleGUI as sg
from tkinter import *
import tkinter as tk


accidentLabels = ['ST_CASE','FATALS','RUR_URB','TYP_INT']
vehicleLabels = ['ST_CASE','DEFORMED','M_HARM','VNUM_LAN','VSPD_LIM','VALIGN', 'VPROFILE','VTRAFWAY']
personLabels = ['ST_CASE','INJ_SEV']
parkworkLabels = ['ST_CASE','PVEH_SEV']
pbtypeLabels = ['ST_CASE','PBCWALK','PBSWALK','PBSZONE']
factorLabels = ['RUR_URB','TYP_INT','VNUM_LAN','VSPD_LIM','VALIGN', 'VPROFILE','VTRAFWAY','PBCWALK','PBSWALK','PBSZONE']
subFactorLabels=['RUR_URB','TYP_INT','VNUM_LAN','VSPD_LIM','VALIGN', 'VPROFILE','VTRAFWAY']



year = ['2015','2016','2017','2018','2019','2020']

def labelNumDict():
    return {'RUR_URB':{'Rural':1,'Urban':2},
                    'TYP_INT':{'N/A':1,'Four-Way':2,'T-Intersection':3,'Y-Intersection':4,'Roundabout':6,'Five Points':7,'L-Intersection':10,'Other':11},
                    'VNUM_LAN':{'Driveway':0,'One Lane': 1,'Two Lanes':2,'Three Lanes':3,'Four Lanes':4,'Five Lanes':5,'Six Lanes':6,'Seven+':7},
                    'VSPD_LIM':{'N/A':0,'25mph':25,'30mph':30,'35mph':35,'40mph':40,'45mph':45,'50mph':50,'55mph':55,'65mph':65,'75mph':75},
                    'VALIGN':{'N/A':0,'Straight':1,'Right Curve':2,'Left Curve':3},
                    'VPROFILE':{'N/A':0,'Level':1,'Slight Grade':2,'Uphill':5,'Downhill':6},
                    'VTRAFWAY':{'N/A':0,'Two-Way, No Division':1,'Two-Way, Divided, No Barrier':2,'Three-Way, Divided, Barrier':3,'One-Way':4,'Two-Way, Continuous Left Turn':5,'Freeway Ramp':6},
                    'PBCWALK':{'No Crosswalk':0,'Crosswalk':1},
                    'PBSWALK':{'No Sidewalk':0,'Sidewalk':1},
                    'PBSZONE':{'No School Zone':0,'School Zone':1},}

def graphOut(xLabel, yLabel, data):
    data.plot.hist(x=xLabel,y=yLabel)
    plt.show()

def plotTraining(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    #ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()

    plt.ylabel('RMSE')
    plt.title('XGBoost Root Mean Square Error')
    plt.show()

def runModel(data,trainLabels,targetLabel):
    x_data = data[trainLabels]
    y_data = data[targetLabel]

    #uncomment to optimize hyperparams
    #data_dmatrix = xgb.DMatrix(data=x_data.values,label=y_data.values)
    #for i in range(1,15):
        #x = i/10
        #params = {"objective": "reg:squarederror", 'learning_rate': x, 'max_depth': 14, 'alpha': 1}
        #print(repr(x))
        #print(paramTest(params,data_dmatrix))

    model = xgb.XGBRegressor(objective='reg:squarederror',learning_rate='0.41',max_depth=14,alpha=1,eval_metric='rmse')

    X_train, X_test, y_train,y_test = train_test_split(x_data.values,y_data.values,test_size = 0.15, random_state = 1234)
    model.fit(X_train,y_train, eval_set=[(X_test,y_test)])
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test,preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    model.save_model('INJURY_model.json')
    print("RMSE: %f" % (rmse))
    print(rmse)
    print("MSE: %f" % (mse))
    print(mse)
    #plotTraining(model)


def paramTest(params,dMat):
    cv_results = xgb.cv(dtrain=dMat, params=params,metrics="rmse", as_pandas=True,seed=1234)
    return cv_results.tail(1)

def train():
    multiYearData = pd.DataFrame()
    for x in range(len(year)):
        save = x!=0

        accident = pd.read_csv(year[x]+'\\accident.csv', encoding='latin1')
        accident = accident[accidentLabels]
        accident.set_index('ST_CASE')

        vehicle = pd.read_csv(year[x]+'\\vehicle.csv', encoding='latin1')
        vehicle = vehicle[vehicleLabels]
        vehicle.set_index('ST_CASE')

        parkwork = pd.read_csv(year[x]+'\\parkwork.csv', encoding='latin1')
        parkwork = parkwork[parkworkLabels]
        parkwork.set_index('ST_CASE')

        person = pd.read_csv(year[x]+'\\person.csv',encoding='latin1')
        person = person[personLabels]
        person.set_index('ST_CASE')


        pbtype = pd.read_csv(year[x]+'\\pbtype.csv',encoding='latin1')
        pbtype = pbtype[pbtypeLabels]
        pbtype.set_index('ST_CASE')


        test = accident.merge(vehicle, how='inner')
        test2 = person.merge(parkwork, how='inner')
        test3 = test.merge(pbtype,how='inner')
        fullData = test3.merge(test2,how='inner')
        multiYearData = pd.concat([multiYearData,fullData])

    runModel(data=multiYearData, trainLabels=factorLabels,targetLabel='INJ_SEV')

def predict():
    diction = labelNumDict()
    Main_Window = tk.Tk()
    Main_Window.title('Road Safety Analysis')
    Main_Window.geometry("300x400")

    modelFatal = xgb.XGBRegressor()
    modelFatal.load_model('FATALS_model.json')

    modelInjury = xgb.XGBRegressor()
    modelInjury.load_model('INJURY_model.json')
    cnt = 0
    dropDowns = []
    dropDownDicts = []
    labels = ["Rural/Urban", "Intersection type", "Lanes", "Speed Limit", "Curves", "Vertical Profile", "Traffic way",
              "Crosswalk", "Sidewalk", "School Zone"]
    for key in diction:
        # Create gas option drop-downs
        options = list(diction.get(key).keys())
        variable = tk.StringVar(Main_Window)
        variable.set(options[0])  # default value
        w = tk.OptionMenu(Main_Window, variable, *options)
        w.grid(column=1, row=cnt + 1)
        label = tk.Label(Main_Window, text=labels[cnt])
        label.grid(column=0, row=cnt + 1)
        cnt = cnt + 1
        dropDowns.append(variable)
        # Get list of the associated numerical values
        dropDownDicts.append(diction.get(key))

    def compute_but():
        values = []
        for i in range(len(dropDowns)):
            values.append(dropDownDicts[i].get(dropDowns[i].get()))

        # WE HAVE A list values that contains all the numbers for the predictor
        roadData = pd.DataFrame(columns=factorLabels)
        roadData.loc[0] = values
        pF = modelFatal.predict(roadData)
        pI = modelInjury.predict(roadData)
        labelF = tk.Label(Main_Window, text="Fatality Score: " + str(pF))
        labelF.grid(row=12, columnspan=2)
        #labelI = tk.Label(Main_Window, text="Injury Score: " + str(pI))
        #labelI.grid(row=13,columnspan=2)
        return values



    button_compute = Button(Main_Window, text='Compute', command=compute_but)
    button_compute.grid(row=11, columnspan=2)


    Main_Window.mainloop()


def main():
    #train()
    predict()

if __name__ == "__main__":
    main()
