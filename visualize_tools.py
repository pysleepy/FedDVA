import torch
import matplotlib.pyplot as plt


def generate_coordinates(n_row, n_col, min_z, max_z, min_c, max_c):
    # generate coordinates
    z_labels = []
    c_labels = []

    z = []
    c = []

    steps_z = (max_z - min_z) / (n_row - 1)
    steps_c = (max_c - min_c) / (n_col - 1)

    for row in range(n_row):
        z_labels.append("{:.2f}".format(min_z + row * steps_z))
        for col in range(n_col):
            z.append([min_z + row * steps_z])
            c.append([min_c + col * steps_c])
            if row == n_row - 1:
                c_labels.append("{:.2f}".format(min_c + col * steps_c))

    z, c = torch.tensor(z), torch.tensor(c)

    return z, c, z_labels, c_labels


def parse_logs(c_id, path_to_log):
    logs = []

    log_name = 'celeba_no_resample_loss_d1'

    with open(log_name) as f:
        for line in f:
            logs.append(line)

    for c_id in range(5):
        # c_id = 4

        dec_c_loss = []
        dec_z_loss = []

        for r, l in enumerate(logs):
            l = l.strip().split(" ")
            print(l)
            if 'Client:' in l and l[3] == str(c_id) + ',':
                if r + 1 >= len(logs):
                    break
                ll = logs[r + 1].split(" ")
                if 'Loss_c:' in ll:
                    dec_c_loss.append(float(ll[-1]))
                if 'Loss_z:' in ll:
                    dec_z_loss.append(float(ll[-1]))

        if len(dec_z_loss) == 0:
            dec_z_loss = dec_c_loss

        dc = []
        dz = []
        mc = 0.
        mz = 0.
        print(len(dec_z_loss))
        print(len(dec_c_loss))
        for r_id, (c, z) in enumerate(zip(dec_c_loss, dec_z_loss)):
            mc += c
            mz += z
            if (r_id + 1) % 5 == 0:
                dc.append(mc / 5.)
                dz.append(mz / 5.)
                mc = 0.
                mz = 0.

        dec_c_loss = dc
        dec_z_loss = dz

        print(len(dec_c_loss))

        plt.figure()
        plt.title(log_name + " " + "Client: {:d}".format(c_id))
        l1, = plt.plot(dec_c_loss, label='dec_c')
        l2, = plt.plot(dec_z_loss, label='dec_z')
        # plt.legend(handles=[l1, l2], labels=['dec_c', 'dec_z'], loc='best')

        plt.legend()
        plt.show()


