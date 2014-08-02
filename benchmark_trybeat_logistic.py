import csv

datareader = csv.reader(open('C:/LearningMaterials/Kaggle/Mlsp/submission_logisticregression.csv','rb'))
header = datareader.next() #skip the first line
open_file_object = csv.writer(open("C:/LearningMaterials/Kaggle/Mlsp/submission_logisticregression_scaledup.csv", "wb"))

cnt = 0
for data in datareader:
    probid = str(data[0])
    prob = float(data[1])

    if prob >= 0.9:
        prob = 1
        cnt += 1

    open_file_object.writerow([probid, prob])

print cnt
print "done."
    
