import time
import torch
import pickle
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from data_generate import Predict_Traj_Dataset, generate_input, generate_input2
from data_genarate3 import generate_input_history, generate_queue, pad_tensor, before_pad_tensor
from bert_language_model import Predict_Model
from bert_traj_model import Bert_Traj_Model
from optimizer import Trans_Optim
from eval_traj import cal_loss_performance, get_acc

def save_model(Epoch, model, file_path="./pretrain/", Predict=False):
    if Predict:
        output_path = file_path + "Predict_model_trained_ep%d.pth" % Epoch
    else:
        output_path = file_path + "Pretrained_ep%d.pth" % Epoch
    # bert_output_path = file_path + "bert_trained_ep%d.pth" % Epoch
    torch.save(model.state_dict(), output_path)
    print("EP:%d Model Saved on:" % Epoch, output_path)
    if not Predict:
        model.save_bert(Epoch)
    print()
    return True

def predict_train_epoch(epoch, model, train_data, train_queue, optimizer, device, batch_size):
    # predict_train_epoch(epoch_i, model, train_data, train_queue, optimizer, device)

    model.train()
    desc= ' -(Train)- '
    total_loss, avg_loss, avg_acc = 0, 0, 0.
    iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time = 0, 0, 0, 0
    total_loc, total_cor_loc = 0, 0
    eva_metric = np.zeros((6, 1))

    len_queue = len(train_queue)
    len_batch = int(np.ceil((len_queue/batch_size)))
    train_queue = [[train_queue.popleft() for _ in range(min(batch_size, len(train_queue)))] for k in range(len_batch)]
    
    for i, batch_queue in enumerate(train_queue):

        max_place = max([len(train_data[u][idx]['loc']) for u,idx in batch_queue]) 
        max_history = max([len(train_data[u][idx]['history_loc']) for u,idx in batch_queue]) 
        
        loc = torch.cat([pad_tensor(train_data[u][idx]['loc'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
        time = torch.cat([pad_tensor(train_data[u][idx]['tim'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
        loc_label = torch.cat([pad_tensor(train_data[u][idx]['target_loc'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long().contiguous().view(-1)
        time_label = torch.cat([pad_tensor(train_data[u][idx]['target_tim'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long().contiguous().view(-1)
        current_itval = torch.cat([pad_tensor(train_data[u][idx]['cur_time_itval'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
           

        history_loc = torch.cat([before_pad_tensor(train_data[u][idx]['history_loc'],max_history,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
        history_time = torch.cat([before_pad_tensor(train_data[u][idx]['history_tim'],max_history,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
        history_itval = torch.cat([before_pad_tensor(train_data[u][idx]['his_time_itval'],max_history,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()    
           

        # his_len_traj = torch.from_numpy(np.array([train_data[u][idx]['his_len_traj'] for u, idx in batch_queue])).to(device).long()

        input_loc = torch.cat([history_loc, loc], dim=1)
        input_tim = torch.cat([history_time, time], dim=1)
        input_itval = torch.cat([current_itval, history_itval], dim=1)

        # print(input_itval)
        # exit()
          
        place_logit, time_logit = model(input_loc, input_tim, max_history, input_itval)

        loss, n_loc, n_cor_loc, n_cor_time = cal_loss_performance(logit1=place_logit, logit2=time_logit, label1=loc_label, label2=time_label, Predict=True)
        eva_metric = get_acc(loc_label, place_logit, eva_metric)

        total_loss += loss.item()
        iter_100_loss += loss.item()
        avg_loss = total_loss/(i+1)

        total_loc += n_loc
        iter_100_loc += n_loc
        total_cor_loc += n_cor_loc
        iter_100_cor_loc += n_cor_loc
        iter_100_cor_time += n_cor_time
        avg_acc = 100.*total_cor_loc/total_loc

        if i % 100 == 0:
            try:
                if n_loc==0 and n_cor_loc==0:
                    n_loc = 1
                print("{} epoch: {:_>2d} | iter: {:_>4d}/{:_>4d} | loss: {:<10.7f} | avg_loss: {:<10.7f} | acc: {:<4.4f} % | avg_acc: {:<4.4f} % | lr: {:<9.7f} | time_acc: {:<4.4f} %".format(
                    desc, epoch, i, len_batch, iter_100_loss/100., avg_loss, 100.*iter_100_cor_loc/iter_100_loc, avg_acc, optimizer._print_lr(), 100.*iter_100_cor_time/iter_100_loc))   
                iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time = 0, 0, 0, 0
            except Exception as e:
                print(e)
                exit()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step_and_update_lr()

    return avg_loss, avg_acc, eva_metric/total_loc

def predict_valid_epoch(epoch, model, valid_data, valid_queue, optimizer, device, batch_size):

    model.eval()
    desc= ' -(valid)- '
    total_loss, avg_loss, avg_acc = 0, 0, 0.
    iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time= 0, 0, 0, 0
    total_loc, total_cor_loc = 0, 0
    eva_metric = np.zeros((6, 1))

    len_queue = len(valid_queue)
    len_batch = int(np.ceil((len_queue/batch_size)))
    valid_queue = [[valid_queue.popleft() for _ in range(min(batch_size, len(valid_queue)))] for k in range(len_batch)]
    
    with torch.no_grad():
        for i, batch_queue in enumerate(valid_queue):

            max_place = max([len(valid_data[u][idx]['loc']) for u,idx in batch_queue]) 
            max_history = max([len(valid_data[u][idx]['history_loc']) for u,idx in batch_queue]) 

            loc = torch.cat([pad_tensor(valid_data[u][idx]['loc'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
            time = torch.cat([pad_tensor(valid_data[u][idx]['tim'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
            loc_label = torch.cat([pad_tensor(valid_data[u][idx]['target_loc'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long().contiguous().view(-1)
            time_label = torch.cat([pad_tensor(valid_data[u][idx]['target_tim'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long().contiguous().view(-1)
            current_itval = torch.cat([pad_tensor(valid_data[u][idx]['cur_time_itval'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
           

            history_loc = torch.cat([before_pad_tensor(valid_data[u][idx]['history_loc'],max_history,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
            history_time = torch.cat([before_pad_tensor(valid_data[u][idx]['history_tim'],max_history,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()    
            history_itval = torch.cat([before_pad_tensor(valid_data[u][idx]['his_time_itval'],max_history,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()    
            


            input_loc = torch.cat([history_loc, loc], dim=1)
            input_tim = torch.cat([history_time, time], dim=1)
            input_itval = torch.cat([current_itval, history_itval], dim=1)

            # print(input_itval)
            # exit()
            
            place_logit, time_logit = model(input_loc, input_tim, max_history, input_itval)

            loss, n_loc, n_cor_loc, n_cor_time = cal_loss_performance(logit1=place_logit, logit2=time_logit, label1=loc_label, label2=time_label, Predict=True)
            eva_metric = get_acc(loc_label, place_logit, eva_metric)

            total_loss += loss.item()
            iter_100_loss += loss.item()
            avg_loss = total_loss/(i+1)

            total_loc += n_loc
            iter_100_loc += n_loc
            total_cor_loc += n_cor_loc
            iter_100_cor_loc += n_cor_loc
            iter_100_cor_time += n_cor_time
            avg_acc = 100.*total_cor_loc/total_loc
            
            if i % 100 == 0:
                print("{} epoch: {:_>2d} | iter: {:_>4d}/{:_>4d} | loss: {:<10.7f} | avg_loss: {:<10.7f} | acc: {:<4.4f} % | avg_acc: {:<4.4f} % | lr: {:<9.7f} | time_acc: {:<4.4f} %".format(
                    desc, epoch, i, len_batch, iter_100_loss/100., avg_loss, 100.*iter_100_cor_loc/iter_100_loc, avg_acc, optimizer._print_lr(), 100.*iter_100_cor_time/iter_100_loc))   
                iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time = 0, 0, 0, 0

    return avg_loss, avg_acc, eva_metric/total_loc


def run(epoch, model, optimizer, device, train_data, train_traj_idxx, valid_data, valid_traj_idxx, log=None, batch_size=4):
    # run(Epoch, model, optimizer, device, train_data, train_traj_idxx, test_data, test_traj_idxx, log=log)

    # with SummaryWriter() as writer:

    log_train_file, log_valid_file = None, None

    if log:
        log_train_file = log + '.train.log'
        log_valid_file = log + '.valid.log'
        # print('[Info] Training performance will be written to file: {} and {}'.format(
        #     log_train_file, log_valid_file))
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write("\n# Note: .\n")
            log_vf.write("\n# Note: .\n")
            log_tf.write("Start Time: {}.\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            log_vf.write("Start Time: {}.\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        
    for epoch_i in range(1, epoch+1):
        
        train_queue = generate_queue(train_traj_idxx,'random','train')
        valid_queue = generate_queue(valid_traj_idxx,'normal','valid') 

        train_avg_loss, train_acc, train_metric = predict_train_epoch(epoch_i, model, train_data, train_queue, optimizer, device, batch_size)
        valid_avg_loss, valid_acc, valid_metric = predict_valid_epoch(epoch_i, model, valid_data, valid_queue, optimizer, device, batch_size)
        print('-'*150)
        print(" --Train--  Epoch: {}/{}  Train_avg_loss: {:<10.7f} Train_acc: {:<4.4f}".format(epoch_i, epoch, train_avg_loss, train_acc))
        print(" --valid--  Epoch: {}/{}  valid_avg_loss: {:<10.7f} valid_acc: {:<4.4f}".format(epoch_i, epoch, valid_avg_loss, valid_acc))
        print('-'*150)
        print(" --Train--  Epoch: {}/{}  Metric: {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f}".format(epoch_i, epoch, train_metric[0][0], train_metric[1][0], train_metric[2][0], train_metric[3][0], train_metric[4][0], train_metric[5][0]))
        print(" --valid--  Epoch: {}/{}  Metric: {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f}".format(epoch_i, epoch, valid_metric[0][0], valid_metric[1][0], valid_metric[2][0], valid_metric[3][0], valid_metric[4][0], valid_metric[5][0]))
        print('-'*150)
        # exit()
        # *********************************************************************
        
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write(" --Train--  Epoch: {}/{}  Train_avg_loss: {} Train_acc: {}\n".format(epoch_i, epoch, train_avg_loss, train_acc))
                log_vf.write(" --valid--  Epoch: {}/{}  valid_avg_loss: {} valid_acc: {}\n".format(epoch_i, epoch, valid_avg_loss, valid_acc)) 
        
        # if epoch_i % 2==0:
        #     save_model(epoch_i, model, Predict=True)
        #     print("The step is {} .".format(optimizer._print_step()))
        #     print('-'*150)
            # writer.add_scalars("Loss", {"Train": train_total_loss, "valid": valid_total_loss}, epoch_i)
            # writer.add_scalars("Acc", {"Train": train_epoch_acc, "valid": valid_epoch_acc}, epoch_i)
            # writer.add_scalars("Lr", {"Train": optimizer._print_lr()}, epoch_i)


def main(Epoch=400, Bert_Pretrain=False, Batch_size=8, Pretrained=False, log='predict'):

    head_n = 12
    d_model = 480
    N_layers = 12
    dropout = 0.4

    # head_n = 10
    # d_model = 400
    # N_layers = 10
    # dropout = 0.1

    print('*'*150)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Get Dataset")
    dataset_4qs = pickle.load(open('./data/tweets-cikm.txttime_interval.pk', 'rb'))

    # print(dataset_4qs.keys())


    # print("exit in main")
    # exit()
    print("User number: ", len(dataset_4qs['uid_list']))
    print("Generate Train_traj_list")
    train_data, train_traj_idxx = generate_input_history(data_neural=dataset_4qs["data_neural"], mode="train")
    print("Generate valid_traj_list")
    test_data, test_traj_idxx = generate_input_history(data_neural=dataset_4qs["data_neural"], mode="test")

    if Bert_Pretrain:
        print("Loaded Pretrained Bert")      
        Bert = Bert_Traj_Model(token_size=len(dataset_4qs['vid_list']), head_n=head_n, d_model=d_model, N_layers=N_layers, dropout=dropout)
        Bert.load_state_dict(torch.load('./pretrain/bert_trained_ep14.pth')) 
    else: 
        print("Create New Bert")      
        Bert = Bert_Traj_Model(token_size=len(dataset_4qs['vid_list']), head_n=head_n, d_model=d_model, N_layers=N_layers, dropout=dropout)

    print("Get Predict Model")
    model = Predict_Model(Bert, token_size=len(dataset_4qs['vid_list']), head_n=head_n, d_model=d_model, N_layers=N_layers, dropout=dropout) 
    if Pretrained:
        print("Load Pretrained Predict Model")
        model.load_state_dict(torch.load('./pretrain/Predict_model_trained_ep34.pth'))
    model = model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    optimizer = Trans_Optim(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        init_lr=3, d_model=500, n_warmup_steps=4000)
    print('*'*150)
    print('-'*65 + "  START TRAIN  " + '-'*65)
    run(Epoch, model, optimizer, device, train_data, train_traj_idxx, test_data, test_traj_idxx, log, Batch_size)

if __name__ == "__main__":
    main()
    pass





