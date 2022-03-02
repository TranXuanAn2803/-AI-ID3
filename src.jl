using Statistics
using RDatasets
using ScikitLearn
using DataFrames
@sk_import metrics : accuracy_score
@sk_import model_selection : train_test_split

iris = dataset("datasets", "iris")


features = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
target = "Species"

x = convert(Array, iris[features])
y = convert(Array, iris[target])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=42)

function entropy(counts, n)
    entropy  = 0
    for i in 1:length(counts)
      res -= (counts[i]*1.0/n) * log2(counts[i]*1.0/n)
    return res
    end
end
function entropy1Division(division)
    n = length(division)
    classes = Set(division)
    
    count=zeros(0)
    for sample in classes
        append!(count,sum(sample==division))
    end
    return entropy(count,sample),n
end
function getEntropy(y_predict, y)
    n = length(y)
    
    EnTrue, nT = entropy_of_one_division(y[y_predict])
    enFalse, nF = entropy_of_one_division(y[~y_predict]) 

    s = nT * 1/n * enTrue + nF * 1/n * enFalse
    return s
end
struct DecisionTreeClassifier
    depth::Int32
    tree::NamedTuple
end
function fit(tree::DecisionTreeClassifier, x, y, node, depth=0)
    if all(x->x==y[1], y)
        node=(val=y[0], index_col= nothing, cutoff=nothing, left=nothing, right=nothing)
        return tree,node
    else
        col_idx, cutoff, entropy = BestSplit(x, y) 
        y_left = y[x[:, col_idx] < cutoff]
        y_right = y[x[:, col_idx] >= cutoff]
        node = (index_col=col_idx,
                cutoff=cutoff,
                val=mean(y),
                left=nothing, 
                right=nothing)
        tree,node.left = fit(tree, X[X[:, col_idx] < cutoff], y_left, nothing, depth+1)
        tree, node.right = fit(tree, X[X[:, col_idx] >= cutoff], y_right, nothing, depth+1)
        tree.depth += 1 
        tree.tree = node
        return tree,node
    end 
end
function BestSplit(tree, x, y)
    idx = nothing
    min = 1
    cutoff = nothing
    i=1
    for (col_data) in eachrow(x.T)
        entropy, cur = findBestSplit(tree,col_data, y)
        if entropy == 0
            return i, cur, entropy
        elseif entropy <= min_entropy
            min = entropy
            idx = i
            cutoff = cur
        end 
        i=i+1
    end
    return idx, cutoff, min
end
function findBestSplit(tree, data, y)       
    min = 10
    for value in Set(data)
        y_predict = data < value
        entropy = getEntropy(y_predict, y)
        if min > entropy
            min= entropy
            cutoff = value
        end 
    end 
    return min_entropy, cutoff
end 

function predict(tree, x)
    t = tree.tree
    pred = zeros(length(x))
    i=1
    for c in eachrow(a):
        pred[i] = predict1(tree, c)
        i=i+1
    end
    return pred
end
function predict1(tree, row):
    cur = tree.tree
    while cur.cutoff
        if row[cur.index_col] < cur.cutoff:
            cur = cur.left
        else:
            cur = cur.right
    end
    return cur.get('val')
end
model = DecisionTreeClassifier()
tree = fit(model,x_train, y_train)
pred=predict(model,x_train)
print('Accuracy of your decision tree model on training data:', accuracy_score(y_train,pred))
pred=predict(model,x_test)
print('Accuracy of your decision tree model:', accuracy_score(y_test,pred))

