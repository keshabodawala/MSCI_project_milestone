import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
import xgboost as xgb
from xgboost import XGBClassifier

from utils.system import parse_params, check_version

import os
#C:\Program Files\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

def generate_features_for_test(stances,dataset,name):
    h, b = [],[]

    for stance in stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X


def get_model():
    return GradientBoostingClassifier(n_estimators=250, random_state=14128, verbose=True)


def save_output(predicted):    
    test_data = pd.read_csv("Data/test_stances.csv")
    test_data['Stance'] = predicted
    test_data.to_csv('answer.csv', index=False, encoding='utf-8') # From pandas library
    

if __name__ == "__main__":
    check_version()
    parse_params()

#    Load the training dataset and generate folds
    training = DataSet()
    validation = DataSet("validation")
    
    training_stances = training.stances
    validation_stances = validation.stances
#    
#    validation = pd.read_csv("Data/test_stances")
    
    X_train,y_train = generate_features(training_stances,training,"train_f1")
#    np.save("features/X_train.npy", X_train)
#    np.save("features/y_train.npy", y_train)
#    
    X_validation,y_validation = generate_features(validation_stances,validation,"validation_f1")
#    np.save("features/X_validation.npy", X_validation)
#    np.save("features/y_validation.npy", y_validation)
#    
    X_total = np.concatenate([X_train, X_validation], axis = 0)
    y_total = np.concatenate([y_train, y_validation], axis = 0)
#    cv_score = 0
#    cv_mf = 0
#    cv_ss = 0
##    for mf in np.arange(1,11,1):
    clf = GradientBoostingClassifier(n_estimators = 200, learning_rate=0.2, min_samples_split=650, min_samples_leaf=50, max_depth=6, max_features='sqrt', subsample=0.8, random_state=10)
    #    param_test1 = {'n_estimators': np.arange(20, 81,10)}
    #    print("Grid Search Started")
    #    gsearch1 = GridSearchCV(estimator = clf, param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=2, verbose = 10)
    clf.fit(X_total,y_total)
    #    
    #    clf = get_model() 
    #    clf.fit(X_train, y_train)
    #    
    #    
#        predicted = [LABELS[int(a)] for a in clf.predict(X_validation)]
#        actual = [LABELS[int(a)] for a in y_validation]
#    
#        score, _ = score_submission(actual, predicted)
#        max_score, _ = score_submission(actual, actual)
#        score = score/max_score
#        
#        print("mf: " + str(mf) + "Score: "+ str(score))
#        if score > cv_score:
#            cv_score = score
#            cv_mf = mf
#            
    
    
    
    
    test = DataSet("test")
    test_stances = test.stances
    X_test = generate_features_for_test(test_stances, test, "test_f3")
#    np.save("features/X_testing.npy", X_test)
#    
    predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
    save_output(predicted)
    
    
    
#    for i in range(test):
#        test[i] = test[i] + predicted[i]
#    print(validation_stances)
#    folds,hold_out = kfold_split(d,n_folds=10)
#    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
#    competition_dataset = DataSet("competition_test")
#    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")
#
#    Xs = dict()
#    ys = dict()
#
#    # Load/Precompute all features now
#    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
#    for fold in fold_stances:
#        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))
#
#
#    best_score = 0
#    best_fold = None
#
#
#    # Classifier for each fold
#    for fold in fold_stances:
#        ids = list(range(len(folds)))
#        del ids[fold]
#
#        X_train = np.vstack(tuple([Xs[i] for i in ids]))
#        y_train = np.hstack(tuple([ys[i] for i in ids]))
#
#        X_test = Xs[fold]
#        y_test = ys[fold]
#
#        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
#        clf.fit(X_train, y_train)
#
#        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
#        actual = [LABELS[int(a)] for a in y_test]
#
#        fold_score, _ = score_submission(actual, predicted)
#        max_fold_score, _ = score_submission(actual, actual)
#
#        score = fold_score/max_fold_score
#
#        print("Score for fold "+ str(fold) + " was - " + str(score))
#        if score > best_score:
#            best_score = score
#            best_fold = clf
#
#
#
#    #Run on Holdout set and report the final score on the holdout set
#    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
#    actual = [LABELS[int(a)] for a in y_holdout]
#
#    print("Scores on the dev set")
#    report_score(actual,predicted)
#    print("")
#    print("")
#
#    #Run on competition dataset
#    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
#    actual = [LABELS[int(a)] for a in y_competition]
#
#    print("Scores on the test set")
#    report_score(actual,predicted)
