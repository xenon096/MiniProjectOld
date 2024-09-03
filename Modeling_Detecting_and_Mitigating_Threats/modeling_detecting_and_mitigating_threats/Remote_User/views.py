from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,detection_type,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('Predict_Threat_Type')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Threat_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            RID = request.POST.get('RID')
            SV_TAXONOMY= request.POST.get('SV_TAXONOMY')
            TAXONOMY_NAME= request.POST.get('TAXONOMY_NAME')
            RG_REFERENCE= request.POST.get('RG_REFERENCE')
            SV_REFERENCE= request.POST.get('SV_REFERENCE')
            SV_NAME= request.POST.get('SV_NAME')
            SV_DESCRIPTION= request.POST.get('SV_DESCRIPTION')
            SL_REFERENCE= request.POST.get('SL_REFERENCE')
            LC_REFERENCE= request.POST.get('LC_REFERENCE')
            PHONE_NUMBER= request.POST.get('PHONE_NUMBER')
            WEBSITE= request.POST.get('WEBSITE')
            EMAIL_ADDRESS= request.POST.get('EMAIL_ADDRESS')
            WHEELCHAIR_ACCESSIBLE= request.POST.get('WHEELCHAIR_ACCESSIBLE')
            STREET_NUMBER= request.POST.get('STREET_NUMBER')
            CITY= request.POST.get('CITY')
            LATITUDE= request.POST.get('LATITUDE')
            LONGITUDE= request.POST.get('LONGITUDE')
            LINK_811= request.POST.get('LINK_811')


        data = pd.read_csv("Healthcare_Datasets.csv", encoding='latin-1')

        def apply_response(Label):
            if (Label == 0):
                return 0  # earthquake
            elif (Label == 1):
                return 1  # explosion

        data['Results'] = data['Label'].apply(apply_response)

        x = data['RID']
        y = data['Results']
        cv = CountVectorizer()

        print(x)
        print(y)

        #x = cv.fit_transform(data['WEBSITE'].apply(lambda x: np.str_(x)))

        x = cv.fit_transform(x)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB

        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))

        print("SGD Classifier")
        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=0)
        sgd_clf.fit(X_train, y_train)
        sgdpredict = sgd_clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, sgdpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, sgdpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, sgdpredict))
        models.append(('SGDClassifier', sgd_clf))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        RID1 = [RID]
        vector1 = cv.transform(RID1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'No Threat'
        elif prediction == 1:
            val = 'Threat'

        print(prediction)
        print(val)

        detection_type.objects.create(
        SV_TAXONOMY=SV_TAXONOMY,
        TAXONOMY_NAME=TAXONOMY_NAME,
        RG_REFERENCE=RG_REFERENCE,
        SV_REFERENCE=SV_REFERENCE,
        SV_NAME=SV_NAME,
        SV_DESCRIPTION=SV_DESCRIPTION,
        SL_REFERENCE=SL_REFERENCE,
        LC_REFERENCE=LC_REFERENCE,
        PHONE_NUMBER=PHONE_NUMBER,
        WEBSITE=WEBSITE,
        EMAIL_ADDRESS=EMAIL_ADDRESS,
        WHEELCHAIR_ACCESSIBLE=WHEELCHAIR_ACCESSIBLE,
        STREET_NUMBER=STREET_NUMBER,
        CITY=CITY,
        LATITUDE=LATITUDE,
        LONGITUDE=LONGITUDE,
        LINK_811=LINK_811,
        Prediction=val)

        return render(request, 'RUser/Predict_Threat_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Threat_Type.html')



