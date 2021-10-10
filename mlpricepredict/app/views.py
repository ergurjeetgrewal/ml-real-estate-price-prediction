import math
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.


def mainpage(request):
    if request.method == 'POST':
        print(request.body)
        CRIM = float(request.POST['CRIM'])
        ZN = float(request.POST['ZN'])
        INDUS = float(request.POST['INDUS'])
        CHAS = request.POST['CHAS']
        if CHAS == 'on':
            CHAS = 1
        else:
            CHAS = 0
        NOX = float(request.POST['NOX'])
        RM = float(request.POST['RM'])
        AGE = float(request.POST['AGE'])
        DIS = float(request.POST['DIS'])
        RAD = float(request.POST['RAD'])
        TAX = float(request.POST['TAX'])
        PTRATIO = float(request.POST['PTRATIO'])
        B = float(request.POST['B'])
        LSTAT = float(request.POST['LSTAT'])
        predata = pricepredictor(CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
                       DIS, RAD, TAX, PTRATIO, B, LSTAT)
        return HttpResponse(round(predata[0],2))
    return render(request, 'app/index.html')


def pricepredictor(*datalist):
    predictiondata = []
    for value in datalist:
        predictiondata.append(value)
    from joblib import load
    model = load(
        'C:/Users/Gurjeet Singh/Desktop/ml frontend/mlpricepredict/app/xyzpvtltd.joblib')
    # features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
    #                       -11.44443979304, -49.31238772,  7.61111401, -26.0016879, -0.5778192,
    #                       -0.97491834,  0.41164221, -66.86091034]])
    features = np.array([predictiondata])
    predata = model.predict(features)
    return predata
