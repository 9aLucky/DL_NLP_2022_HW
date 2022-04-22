import random

theta = {'s1':0.2, 's2':0.5, 'p': 0.8, 'q': 0.2, 'r': 0.5}
theta_init = {'s1':0.8, 's2':0.1, 'p': 0.7, 'q': 0.3, 'r': 0.4}
theta_record = {'s1':[0.8], 's2':[0.1], 'p': [0.7], 'q': [0.3], 'r': [0.4]}
times = 10000
x_seq = []
eps = 0.001
group_size = 20


def generate_res():
    for i in range(times):
        xx_seq = []
        rand_value = random.random()
        for k in range(group_size):
            if rand_value < theta['s1']:
                tmp = 1 if random.random() > theta['p'] else 0
            elif rand_value < theta['s1'] + theta['s2']:
                tmp = 1 if random.random() > theta['q'] else 0
            else:
                tmp = 1 if random.random() > theta['r'] else 0
            xx_seq.append(tmp)
        x_seq.append(xx_seq)


def record():
    theta_record['s1'].append(theta_init['s1'])
    theta_record['s2'].append(theta_init['s2'])
    theta_record['p'].append(theta_init['p'])
    theta_record['q'].append(theta_init['q'])
    theta_record['r'].append(theta_init['r'])


def em():
    p1i_record = []
    p2i_record = []
    p3i_record = []
    for xi in x_seq:
        real_xi = sum(xi)
        real_neg_xi = len(xi) - real_xi
        p = theta_init['p']
        q = theta_init['q']
        r = theta_init['r']
        s1 = theta_init['s1']
        s2 = theta_init['s2']
        s3 = 1 - theta_init['s1'] - theta_init['s2']
        pA = s1 * pow(p, real_xi) * pow(1-p, real_neg_xi)
        pB = s2 * pow(q, real_xi) * pow(1-q, real_neg_xi)
        pC = s3 * pow(r, real_xi) * pow(1-r, real_neg_xi)
        p1i = pA / (pA + pB + pC)
        p2i = pB / (pA + pB + pC)
        p3i = pC / (pA + pB + pC)
        p1i_record.append(p1i)
        p2i_record.append(p2i)
        p3i_record.append(p3i)
    theta_init['s1'] = sum(p1i_record)/len(x_seq)
    theta_init['s2'] = sum(p2i_record)/len(x_seq)
    theta_init['p'] = sum(p1i_record[i]*sum(x_seq[i])/len(x_seq[i]) for i in range(len(x_seq))) / sum(p1i_record)
    theta_init['q'] = sum(p2i_record[i]*sum(x_seq[i])/len(x_seq[i]) for i in range(len(x_seq))) / sum(p2i_record)
    theta_init['r'] = sum(p3i_record[i]*sum(x_seq[i])/len(x_seq[i]) for i in range(len(x_seq))) / sum(p3i_record)
    # theta_init['q'] = sum((p2i_record[i]*x_seq[i]) for i in range(len(x_seq))) / sum(p2i_record)
    # theta_init['r'] = sum((p3i_record[i]*x_seq[i]) for i in range(len(x_seq))) / sum(p3i_record)
    record()


if __name__ == '__main__':
    generate_res()
    for i in range(20):
        em()

        p = theta_record['p'][-2]
        q = theta_record['q'][-2]
        r = theta_record['r'][-2]
        s1 = theta_record['s1'][-2]
        s2 = theta_record['s2'][-2]

        p_new = theta_record['p'][-1]
        q_new = theta_record['q'][-1]
        r_new = theta_record['r'][-1]
        s1_new = theta_record['s1'][-1]
        s2_new = theta_record['s2'][-1]

        # if abs(p-p_new) < eps * abs(q-q_new) < eps * abs(r-r_new) < eps \
        #         * abs(s1-s1_new) < eps * abs(s2-s2_new) < eps:
        #     break

    print(str.format('s1: {:.4f}', theta_record['s1'][-1]))
    print(str.format('s2: {:.4f}', theta_record['s2'][-1]))
    print(str.format('p: {:.4f}', theta_record['p'][-1]))
    print(str.format('q: {:.4f}', theta_record['q'][-1]))
    print(str.format('r: {:.4f}', theta_record['r'][-1]))
    print(theta_record['s1'])
    print(theta_record['s2'])
    print(theta_record['p'])
    print(theta_record['q'])
    print(theta_record['r'])

