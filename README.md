# learn_decision_tree

决策树的要点：  
在构建一个决策树时，最重要的是需要确定每一步将哪个特征作为节点。  
原则：  
在每一次决定构建新节点时，对每个特征都进行如下计算：  
    
    如果将该特征作为节点，计算熵增益值。  
然后，选择熵增益最大的特征，作为当前新建的节点。  
直到所有特征都已经用上。  
