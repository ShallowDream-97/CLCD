import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion import Fusion,Fusion_Ori
import dgl
import numpy as np

class Net(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256
        #self.directed_g = remove_random_edges(local_map['directed_g'],0.05)
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.u_from_e = local_map['u_from_e'].to(self.device)
        self.e_from_u = local_map['e_from_u'].to(self.device)

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.knowledge_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.k_index = torch.LongTensor(list(range(self.stu_dim))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)

        
        if args.id == 0:
            self.FusionLayer1 = Fusion_Ori(args, local_map)
            self.FusionLayer2 = Fusion_Ori(args, local_map)
        elif args.id == 1:
            self.FusionLayer1 = Fusion(args, local_map)
            self.FusionLayer2 = Fusion_Ori(args, local_map)
        else:
            self.FusionLayer1 = Fusion_Ori(args, local_map)
            self.FusionLayer2 = Fusion(args, local_map)

        self.prednet_full1 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        # self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        # self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * args.knowledge_n, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_r,args):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)
        
        
        #graph_pertuation(这个位置发生图扰动)
        if args.id == 1:
        # Fusion layer 1
            kn_emb1, exer_emb1, all_stu_emb1,all_stu_emb1_per = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
            # Fusion layer 2
            kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)
            kn_emb2, exer_emb2, all_stu_emb2_per = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1_per)
        # get batch student data
            batch_stu_emb = all_stu_emb2[stu_id] # 32 123
            batch_stu_emb_per = all_stu_emb2_per[stu_id] # 32 123
            
            batch_stu_emb_normalized = torch.nn.functional.normalize(batch_stu_emb, p=2, dim=1)
            batch_stu_emb_1_normalized = torch.nn.functional.normalize(batch_stu_emb_per, p=2, dim=1)
            # 计算余弦相似度
            print("batch_stu_emb_normalized")
            print(batch_stu_emb_normalized)
            print("batch_stu_emb_1_normalized")
            print(batch_stu_emb_1_normalized)
            
            similarity = torch.matmul(batch_stu_emb_normalized, batch_stu_emb_1_normalized.T)
                # 计算范数矩阵
            norm_matrix = torch.sqrt(torch.sum(batch_stu_emb_normalized ** 2, dim=1, keepdim=True))
            norm_matrix = torch.matmul(norm_matrix, norm_matrix.t())

            # 计算余弦相似度矩阵
            similarity = similarity / norm_matrix
            # for u in (range(batch_stu_emb.shape[0])):
            #     similarity = cosine_similarity(batch_stu_emb_normalized, batch_stu_emb_1_normalized.T)
            soft_para = 0.2
            similarity = similarity/soft_para
            # 应用 exp 函数
            exp_similarity = torch.exp(similarity)
            # 计算相同索引行的余弦相似度
            diag_similarities = exp_similarity.diag()
            # 初始化总损失
            ssl_loss = 0.0
            # print("Compute Sim in the batch:")
            for u in (range(batch_stu_emb.shape[0])):
                # 获取当前行的余弦相似度
                u_row_similarity = similarity[u, :]
                u_u_similarity = diag_similarities[u]
                # 计算与当前行的余弦相似度之和
                sum_similarity = torch.sum(u_row_similarity)

                # 除以余弦相似度之和
                divided_similarity = u_u_similarity / (sum_similarity + 1e-8)
                # 计算当前行的损失
                row_loss = torch.sum(divided_similarity)
                # print(type(row_loss))
                row_loss =-torch.log(torch.clamp(row_loss, min=1e-8))
                # 累积到总损失
                ssl_loss += row_loss
                # print(time.time()-similarity_com_bef)
                # print("ssl_loss:"+str(ssl_loss))
            #ssl_loss = ssl_loss/batch_stu_emb.shape[0]
            
        
        elif args.id == 0:
            kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
            # Fusion layer 2
            kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)

        # get batch student data
            batch_stu_emb = all_stu_emb2[stu_id] # 32 123
            ssl_loss = 0.0
            
        #graph_pertuation(这个位置发生图扰动)
        else:
        # Fusion layer 1
            kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
            # Fusion layer 2
            kn_emb2, exer_emb2, all_stu_emb2,all_stu_emb2_per = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)
        # get batch student data
            batch_stu_emb = all_stu_emb2[stu_id] # 32 123
            batch_stu_emb_per = all_stu_emb2_per[stu_id] # 32 123
            
            batch_stu_emb_normalized = torch.nn.functional.normalize(batch_stu_emb, p=2, dim=1)
            batch_stu_emb_1_normalized = torch.nn.functional.normalize(batch_stu_emb_per, p=2, dim=1)
            # 计算余弦相似度
            similarity = torch.matmul(batch_stu_emb_normalized, batch_stu_emb_1_normalized.T)
                # 计算范数矩阵
            norm_matrix = torch.sqrt(torch.sum(batch_stu_emb_normalized ** 2, dim=1, keepdim=True))
            norm_matrix = torch.matmul(norm_matrix, norm_matrix.t())

            # 计算余弦相似度矩阵
            similarity = similarity / norm_matrix
            # for u in (range(batch_stu_emb.shape[0])):
            #     similarity = cosine_similarity(batch_stu_emb_normalized, batch_stu_emb_1_normalized.T)
            soft_para = 0.2
            similarity = similarity/soft_para
            # 应用 exp 函数
            exp_similarity = torch.exp(similarity)
            # 计算相同索引行的余弦相似度
            diag_similarities = exp_similarity.diag()
            # 初始化总损失
            ssl_loss = 0.0
            # print("Compute Sim in the batch:")
            for u in (range(batch_stu_emb.shape[0])):
                # 获取当前行的余弦相似度
                u_row_similarity = similarity[u, :]
                u_u_similarity = diag_similarities[u]
                # 计算与当前行的余弦相似度之和
                sum_similarity = torch.sum(u_row_similarity)

                # 除以余弦相似度之和
                divided_similarity = u_u_similarity / (sum_similarity + 1e-8)
                # 计算当前行的损失
                row_loss = torch.sum(divided_similarity)
                # print(type(row_loss))
                row_loss =-torch.log(torch.clamp(row_loss, min=1e-8))
                # 累积到总损失
                ssl_loss += row_loss
                # print(time.time()-similarity_com_bef)
                # print("ssl_loss:"+str(ssl_loss))
            #ssl_loss = ssl_loss/batch_stu_emb.shape[0]
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1])

        # get batch exercise data
        batch_exer_emb = exer_emb2[exer_id]  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb2.shape[0], kn_emb2.shape[1])

        # Cognitive diagnosis
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim = 1)
        count_of_concept = torch.sum(kn_r, dim = 1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output,ssl_loss

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
