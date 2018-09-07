import logging
import data
from CDL import CollaborativeDeepLearning
import numpy as np
import scipy.sparse as sparse

def main():
    logging.info('reading data')

    item_mat = data.get_mult()

    trainM = sparse.csr_matrix(data.read_user(f_in='data/dummy/cf-train-10-users.dat',num_u=50,num_v=1929))
    testM= sparse.csr_matrix(data.read_user(f_in='data/dummy/cf-test-10-users.dat',num_u=50,num_v=1929))

    trainList = list()
    testList = list()
    for user in range(trainM.shape[0]):
        negative = 0
        for item in range(trainM.shape[1]):
            if trainM[user, item] == 1:
                trainList.append( [user, item, 1] )     
            else: 
                if negative < 20:
                    trainList.append( [user, item, 0] )
                    negative+=1        
        train = np.array(trainList).astype('float32')

    testList = list()
    for user in range(testM.shape[0]):
        negative = 0
        for item in range(testM.shape[1]):
            if testM[user, item] == 1:
                testList.append( [user, item, 1] )         
    #        else:
    #            if negative < 10:
    #                testList.append( [user, item, 0] )
    #                negative+=1                 
        test = np.array(testList).astype('float32')


    num_item_feat = item_mat.shape[1]

    model = CollaborativeDeepLearning(item_mat, [num_item_feat, 50, 10])
    model.pretrain(lamda_w=0.001, encoder_noise=0.3, epochs=10)
    model_history = model.fineture(train, test, lamda_u=0.01, lamda_v=0.1, lamda_n=0.1, lr=0.01, epochs=500)
    testing_rmse = model.getRMSE(test)
    print('Testing RMSE = {}'.format(testing_rmse))
    
    import metrics
    print('AUC %s' % metrics.full_auc(model.cdl_model, testM))
    
    import matplotlib.pyplot as plt
    M_low = 50
    M_high = 300
    recall_levels = M_high-M_low + 1
    recallArray = np.zeros(6)
    x=0
    for n in [50, 100, 150, 200, 250, 300]:
        test_recall = metrics.recall_at_k(model.cdl_model, testM, k=n)
        recallArray[x] = test_recall
        print('Recall: %.2f.' % (test_recall))
        x+=1
    plt.plot([50, 100, 150, 200, 250, 300],recallArray)
    plt.ylabel("Recall")
    plt.xlabel("M")
    plt.title("Proposed: Recall@M")
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    main()