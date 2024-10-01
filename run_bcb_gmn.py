import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm, trange
import models
import os
import javalang
import javalang.tree
import javalang.ast
import javalang.util
from javalang.ast import Node
from anytree import AnyNode
from torch_geometric import data


class RunBcbGMN:
    def __init__(self):
        self.edges = {
            "Nexttoken": 2,
            "Prevtoken": 3,
            "Nextuse": 4,
            "Prevuse": 5,
            "If": 6,
            "Ifelse": 7,
            "While": 8,
            "For": 9,
            "Nextstmt": 10,
            "Prevstmt": 11,
            "Prevsib": 12,
        }
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--cuda", default=True)
        self.parser.add_argument("--dataset", default="gcj")
        self.parser.add_argument("--graph_mode", default="ast_and_next")
        self.parser.add_argument("--next_sib", default=False)
        self.parser.add_argument("--if_edge", default=False)
        self.parser.add_argument("--while_edge", default=False)
        self.parser.add_argument("--for_edge", default=False)
        self.parser.add_argument("--block_edge", default=False)
        self.parser.add_argument("--next_token", default=False)
        self.parser.add_argument("--next_use", default=False)
        self.parser.add_argument("--data_setting", default="11")
        self.parser.add_argument("--batch_size", default=32)
        self.parser.add_argument("--num_layers", default=4)
        self.parser.add_argument("--num_epochs", default=3)
        self.parser.add_argument("--lr", default=0.001)
        self.parser.add_argument("--threshold", default=0)
        self.args = self.parser.parse_args()
        self.device = (
            torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("CUDA", torch.cuda.is_available())
        self.ast_dict, self.vocab_len, self.vocab_dict = self.create_ast()
        self.tree_dict = self.create_separate_graph(
            self.ast_dict,
            self.vocab_len,
            self.vocab_dict,
            self.device,
            mode=self.args.graph_mode,
            next_sib=self.args.next_sib,
            if_edge=self.args.if_edge,
            while_edge=self.args.while_edge,
            for_edge=self.args.for_edge,
            block_edge=self.args.block_edge,
            next_token=self.args.next_token,
            next_use=self.args.next_use,
        )
        self.train_data, self.valid_data, self.test_data = self.create_gmn_data(
            self.args.data_setting,
            self.tree_dict,
            self.vocab_len,
            self.vocab_dict,
            self.device,
        )
        print(len(self.train_data))
        num_layers = int(self.args.num_layers)
        self.model = models.GMNnet(
            self.vocab_len, embedding_dim=100, num_layers=num_layers, device=self.device
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CosineEmbeddingLoss()
        self.criterion_2 = nn.MSELoss()

    def get_edge_next_sib(self, node, vocab_dict, src, tgt, edge_type):
        for i in range(len(node.children) - 1):
            src.append(node.children[i].id)
            tgt.append(node.children[i + 1].id)
            edge_type.append([1])
            src.append(node.children[i + 1].id)
            tgt.append(node.children[i].id)
            edge_type.append([self.edges["Prevsib"]])
        for child in node.children:
            self.get_edge_next_sib(child, vocab_dict, src, tgt, edge_type)

    def edge_flow_append(self, edge, node, src, tgt, edge_type):
        src.append(node.children[0].id)
        tgt.append(node.children[1].id)
        edge_type.append([self.edges[edge]])
        src.append(node.children[1].id)
        tgt.append(node.children[0].id)
        edge_type.append([self.edges[edge]])

    def get_edge_flow(
        self,
        node,
        vocab_dict,
        src,
        tgt,
        edge_type,
        if_edge=False,
        while_edge=False,
        for_edge=False,
    ):
        token = node.token
        if while_edge == True and token == "WhileStatement":
            self.edge_flow_append("While", node, src, tgt, edge_type)
        if for_edge == True and token == "ForStatement":
            self.edge_flow_append("For", node, src, tgt, edge_type)
        if if_edge == True and token == "IfStatement":
            self.edge_flow_append("If", node, src, tgt, edge_type)
            if len(node.children) == 3:
                src.append(node.children[0].id)
                tgt.append(node.children[2].id)
                edge_type.append([self.edges["Ifelse"]])
                src.append(node.children[2].id)
                tgt.append(node.children[0].id)
                edge_type.append([self.edges["Ifelse"]])
        for child in node.children:
            self.get_edge_flow(
                child, vocab_dict, src, tgt, edge_type, if_edge, while_edge, for_edge
            )

    def get_edge_next_stmt(self, node, vocab_dict, src, tgt, edge_type):
        token = node.token
        if token == "BlockStatement":
            for i in range(len(node.children) - 1):
                src.append(node.children[i].id)
                tgt.append(node.children[i + 1].id)
                edge_type.append([self.edges["Nextstmt"]])
                src.append(node.children[i + 1].id)
                tgt.append(node.children[i].id)
                edge_type.append([self.edges["Prevstmt"]])
        for child in node.children:
            self.get_edge_next_stmt(child, vocab_dict, src, tgt, edge_type)

    def get_edge_next_token(self, node, vocab_dict, src, tgt, edge_type, token_list):
        def get_token_list(node, vocab_dict, edge_type, token_list):
            if len(node.children) == 0:
                token_list.append(node.id)
            for child in node.children:
                get_token_list(child, vocab_dict, edge_type, token_list)

        get_token_list(node, vocab_dict, edge_type, token_list)
        for i in range(len(token_list) - 1):
            src.append(token_list[i])
            tgt.append(token_list[i + 1])
            edge_type.append([self.edges["Nexttoken"]])
            src.append(token_list[i + 1])
            tgt.append(token_list[i])
            edge_type.append([self.edges["Prevtoken"]])

    def get_edge_next_use(self, node, vocab_dict, src, tgt, edge_type, variable_dict):
        def get_variables(node, vocab_dict, edge_type, variable_dict):
            token = node.token
            if token == "MemberReference":
                for child in node.children:
                    if child.token == node.data.member:
                        variable = child.token
                        variable_node = child
                if not variable_dict.__contains__(variable):
                    variable_dict[variable] = [variable_node.id]
                else:
                    variable_dict[variable].append(variable_node.id)
            for child in node.children:
                get_variables(child, vocab_dict, edge_type, variable_dict)

        get_variables(node, vocab_dict, edge_type, variable_dict)
        for v in variable_dict.keys():
            for i in range(len(variable_dict[v]) - 1):
                src.append(variable_dict[v][i])
                tgt.append(variable_dict[v][i + 1])
                edge_type.append([self.edges["Nextuse"]])
                src.append(variable_dict[v][i + 1])
                tgt.append(variable_dict[v][i])
                edge_type.append([self.edges["Prevuse"]])

    def get_token(self, node):
        token = ""
        if isinstance(node, str):
            token = node
        elif isinstance(node, set):
            token = "Modifier"
        elif isinstance(node, Node):
            token = node.__class__.__name__
        return token

    def get_child(self, root):
        if isinstance(root, Node):
            children = root.children
        elif isinstance(root, set):
            children = list(root)
        else:
            children = []

        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        yield sub_item
                elif item:
                    yield item

        return list(expand(children))

    def get_sequence(self, node, sequence):
        token, children = self.get_token(node), self.get_child(node)
        sequence.append(token)
        for child in children:
            self.get_sequence(child, sequence)

    def get_nodes(self, node, node_list):
        node_list.append(node)
        children = self.get_child(node)
        for child in children:
            self.get_nodes(child, node_list)

    def create_tree(self, root, node, node_list, parent=None):
        id = len(node_list)
        token, children = self.get_token(node), self.get_child(node)
        if id == 0:
            root.token = token
            root.data = node
        else:
            new_node = AnyNode(id=id, token=token, data=node, parent=parent)
        node_list.append(node)
        for child in children:
            if id == 0:
                self.create_tree(root, child, node_list, parent=root)
            else:
                self.create_tree(root, child, node_list, parent=new_node)

    def get_node_and_edge_ast_only(self, node, node_index_list, vocab_dict, src, tgt):
        token = node.token
        node_index_list.append([vocab_dict[token]])
        for child in node.children:
            src.append(node.id)
            tgt.append(child.id)
            src.append(child.id)
            tgt.append(node.id)
            self.get_node_and_edge_ast_only(
                child, node_index_list, vocab_dict, src, tgt
            )

    def get_node_and_edge(self, node, node_index_list, vocab_dict, src, tgt, edge_type):
        token = node.token
        node_index_list.append([vocab_dict[token]])
        for child in node.children:
            src.append(node.id)
            tgt.append(child.id)
            edge_type.append([0])
            src.append(child.id)
            tgt.append(node.id)
            edge_type.append([0])
            self.get_node_and_edge(
                child, node_index_list, vocab_dict, src, tgt, edge_type
            )

    def count_nodes(self, node, if_count, while_count, for_count, block_count):
        token = node.token
        if token == "IfStatement":
            if_count += 1
        if token == "WhileStatement":
            while_count += 1
        if token == "ForStatement":
            for_count += 1
        if token == "BlockStatement":
            block_count += 1
        print(if_count, while_count, for_count, block_count)
        for child in node.children:
            self.count_nodes(child, if_count, while_count, for_count, block_count)

    def create_ast(self):
        asts, paths, all_tokens = [], [], []
        dir_name = "BCB/bigclonebenchdata/"
        for rt, _dirs, files in os.walk(dir_name):
            for file in files:
                program_file = open(os.path.join(rt, file), encoding="utf-8")
                program_text = program_file.read()
                program_tokens = javalang.tokenizer.tokenize(program_text)
                parser = javalang.parse.Parser(program_tokens)
                program_ast = parser.parse_member_declaration()
                paths.append(os.path.join(rt, file))
                asts.append(program_ast)
                self.get_sequence(program_ast, all_tokens)
                program_file.close()
        ast_dict = dict(zip(paths, asts))
        if_count, while_count, for_count, block_count, do_count, switch_count = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
        for token in all_tokens:
            if token == "IfStatement":
                if_count += 1
            if token == "WhileStatement":
                while_count += 1
            if token == "ForStatement":
                for_count += 1
            if token == "BlockStatement":
                block_count += 1
            if token == "DoStatement":
                do_count += 1
            if token == "SwitchStatement":
                switch_count += 1
        print(if_count, while_count, for_count, block_count, do_count, switch_count)
        print("all_nodes ", len(all_tokens))
        all_tokens = list(set(all_tokens))
        vocab_size = len(all_tokens)
        token_ids = range(vocab_size)
        vocab_dict = dict(zip(all_tokens, token_ids))
        print(vocab_size)
        return ast_dict, vocab_size, vocab_dict

    def create_separate_graph(
        self,
        ast_dict,
        vocab_len,
        vocab_dict,
        device,
        mode="ast_only",
        next_sib=False,
        if_edge=False,
        while_edge=False,
        for_edge=False,
        block_edge=False,
        next_token=False,
        next_use=False,
    ):
        path_list, tree_list = [], []
        print("next_sib ", next_sib)
        print("if_edge ", if_edge)
        print("while_edge ", while_edge)
        print("for_edge ", for_edge)
        print("block_edge ", block_edge)
        print("next_token", next_token)
        print("next_use ", next_use)
        print(len(ast_dict))
        for path, tree in ast_dict.items():
            node_list = []
            new_tree = AnyNode(id=0, token=None, data=None)
            self.create_tree(new_tree, tree, node_list)
            x, edge_src, edge_tgt, edge_attr = [], [], [], []
            if mode == "ast_only":
                self.get_node_and_edge_ast_only(
                    new_tree, x, vocab_dict, edge_src, edge_tgt
                )
            else:
                self.get_node_and_edge(
                    new_tree, x, vocab_dict, edge_src, edge_tgt, edge_attr
                )
                if next_sib == True:
                    self.get_edge_next_sib(
                        new_tree, vocab_dict, edge_src, edge_tgt, edge_attr
                    )
                self.get_edge_flow(
                    new_tree,
                    vocab_dict,
                    edge_src,
                    edge_tgt,
                    edge_attr,
                    if_edge,
                    while_edge,
                    for_edge,
                )
                if block_edge == True:
                    self.get_edge_next_stmt(
                        new_tree, vocab_dict, edge_src, edge_tgt, edge_attr
                    )
                tokenlist = []
                if next_token == True:
                    self.get_edge_next_token(
                        new_tree, vocab_dict, edge_src, edge_tgt, edge_attr, tokenlist
                    )
                variable_dict = {}
                if next_use == True:
                    self.get_edge_next_use(
                        new_tree,
                        vocab_dict,
                        edge_src,
                        edge_tgt,
                        edge_attr,
                        variable_dict,
                    )
            edge_index = [edge_src, edge_tgt]
            ast_length = len(x)
            path_list.append(path)
            tree_list.append([[x, edge_index, edge_attr], ast_length])
            ast_dict[path] = [[x, edge_index, edge_attr], ast_length]
        return ast_dict

    def create_gmn_data(self, id, tree_dict, vocab_len, vocab_dict, device):
        index_dir = "BCB/"
        if id == "0" or id == "11":
            train_file, valid_file, test_file = (
                #open(index_dir + ("traindata.txt" if id == "0" else "traindata11.txt")),
                open(index_dir + "traindata11big.txt"),
                open(index_dir + "devdata.txt"),
                open(index_dir + "testdata.txt"),
            )
        else:
            print("file not exist")
            quit()
        train_list, valid_list, test_list = (
            train_file.readlines(),
            valid_file.readlines(),
            test_file.readlines(),
        )
        train_data, valid_data, test_data = [], [], []
        print("train_data")
        train_data = self.create_pair_data(tree_dict, train_list, device=device)
        print("valid_data")
        valid_data = self.create_pair_data(tree_dict, valid_list, device=device)
        print("test_data")
        test_data = self.create_pair_data(tree_dict, test_list, device=device)
        return train_data, valid_data, test_data

    def create_pair_data(self, tree_dict, path_list, device):
        data_list = []
        count_lines = 1
        for line in path_list:
            count_lines += 1
            pair_info = line.split()
            code_1_path = "BCB" + pair_info[0].strip(".")
            code_2_path = "BCB" + pair_info[1].strip(".")
            label = torch.tensor(int(pair_info[2]), dtype=torch.float, device=self.device)
            data_1 = tree_dict[code_1_path]
            data_2 = tree_dict[code_2_path]
            x_1, edge_index_1, edge_attr_1, ast_1_length = (
                data_1[0][0],
                data_1[0][1],
                data_1[0][2],
                data_1[1],
            )
            x_2, edge_index_2, edge_attr_2, ast_2_length = (
                data_2[0][0],
                data_2[0][1],
                data_2[0][2],
                data_2[1],
            )
            if edge_attr_1 == []:
                edge_attr_1 = None
                edge_attr_2 = None
            data = [
                [x_1, x_2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2],
                label,
            ]
            data_list.append(data)
        return data_list

    def create_batches(self, data):
        batches = [
            data[graph : graph + self.args.batch_size]
            for graph in range(0, len(data), self.args.batch_size)
        ]
        return batches

    def predict(self, data):
        x_1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2 = data
        x_1 = torch.tensor(x_1, dtype=torch.long, device=self.device)
        x2 = torch.tensor(x2, dtype=torch.long, device=self.device)
        edge_index_1 = torch.tensor(edge_index_1, dtype=torch.long, device=self.device)
        edge_index_2 = torch.tensor(edge_index_2, dtype=torch.long, device=self.device)
        if edge_attr_1 != None:
            edge_attr_1 = torch.tensor(
                edge_attr_1, dtype=torch.long, device=self.device
            )
            edge_attr_2 = torch.tensor(
                edge_attr_2, dtype=torch.long, device=self.device
            )
        data = [x_1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2]
        return self.model(data)

    def validate(self, dataset):
        tp, tn, fp, fn = 0, 0, 0, 0
        results = []
        for data, label in dataset:
            prediction = self.predict(data)
            output = F.cosine_similarity(prediction[0], prediction[1])
            results.append(output.item())
            prediction = torch.sign(output).item()
            if prediction > self.args.threshold and label.item() == 1:
                tp += 1
            if prediction <= self.args.threshold and label.item() == -1:
                tn += 1
            if prediction > self.args.threshold and label.item() == -1:
                fp += 1
            if prediction <= self.args.threshold and label.item() == 1:
                fn += 1
        print(tp, tn, fp, fn)
        p, r, f1 = 0.0, 0.0, 0.0
        if tp + fp == 0:
            print("precision is none")
            return results
        p = tp / (tp + fp)
        if tp + fn == 0:
            print("recall is none")
            return results
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        print("precision", p)
        print("recall", r)
        print("F1", f1)
        return results

    def run(self):
        epochs = trange(self.args.num_epochs, leave=True, desc="Epoch")
        for epoch in epochs:  # without batching
            print(epoch)
            batches = self.create_batches(self.train_data)
            total_loss = 0.0
            main_index = 0.0
            for index, batch in tqdm(
                enumerate(batches), total=len(batches), desc="Batches"
            ):
                self.optimizer.zero_grad()
                batch_loss = 0
                for data, label in batch:
                    prediction = self.predict(data)
                    cos_sim = F.cosine_similarity(prediction[0], prediction[1])
                    batch_loss += self.criterion_2(cos_sim, label)
                batch_loss.backward(retain_graph=True)
                self.optimizer.step()
                loss = batch_loss.item()
                total_loss += loss
                main_index = main_index + len(batch)
                loss = total_loss / main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            print('validation...')
            self.dev_results = self.validate(self.valid_data)
            print('validation done')
            print('validation res file opening...')
            dev_file = open(
                "gmnbcbresult/" + self.args.graph_mode + "_dev_epoch_" + str(epoch + 1),
                mode="w",
            )
            print('write validation results...')
            for res in self.dev_results:
                dev_file.write(str(res) + "\n")
            print('writing validation results done')
            dev_file.close()
            print('testing...')
            self.test_results = self.validate(self.test_data)
            print('testing done')
            print('test res file opening...')
            res_file = open(
                "gmnbcbresult/" + self.args.graph_mode + "_epoch_" + str(epoch + 1),
                mode="w",
            )
            print('write test results...')
            for res in self.test_results:
                res_file.write(str(res) + "\n")
            print('writing test results done')
            res_file.close()

            print('saving model...')
            torch.save(self.model.state_dict(), "gmnmodels/gmnbcb" + str(epoch + 1))
            print('saving done')


if __name__ == "__main__":
    gmn_run = RunBcbGMN()
    gmn_run.run()
