import numpy as np
def get_batches(pairs, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size#先判断有多少个batch数

    for idx in range(n_batches):#遍历每一个batch的元素
        edges, label = [], []
        for i in range(batch_size):#利用batch_size实现上述目的
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            edges.append(pairs[index, 0:2])
            label.append(pairs[index][2])
        yield np.array(edges), np.array(label)

def get_mask2index(mask, label=3): # 3-unknown
    nonzero_indices = np.nonzero(mask)
    unlabel = np.ones_like(nonzero_indices[0]) * 3
    unlabel_pair = np.column_stack((nonzero_indices[0], nonzero_indices[1], unlabel))
    return unlabel_pair, len(unlabel_pair)




if __name__ == '__main__':
    print('this is example')
