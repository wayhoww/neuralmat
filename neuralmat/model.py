import torch


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out, n_features_hidden=64, n_hidden_layers=3):
        super().__init__()

        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.n_features_hidden = n_features_hidden
        self.n_hidden_layers = n_hidden_layers

        if self.n_hidden_layers == 0:
            self._fc = torch.nn.ModuleList(
                [torch.nn.Linear(self.n_features_in, self.n_features_out)]
            )
        else:
            self._fc = torch.nn.ModuleList(
                [torch.nn.Linear(self.n_features_in, self.n_features_hidden)] + 
                [torch.nn.Linear(self.n_features_hidden, self.n_features_hidden) for _ in range(self.n_hidden_layers - 1)] + 
                [torch.nn.Linear(self.n_features_hidden, self.n_features_out)]
            )

    def forward(self, x) -> torch.Tensor:
        activate = torch.relu

        for i in range(self.n_hidden_layers):
            x = activate(self._fc[i](x))
        x = self._fc[self.n_hidden_layers](x)
        return x

    def generate_weight(self, layer: torch.nn.Linear):
        weight = layer.weight.data.cpu().detach().numpy()
        out = '{'
        for i in range(weight.shape[0]):
            out += '{'
            for j in range(weight.shape[1]):
                out += f"{weight[i, j]:.7f}, "
            out += '}, '
        out += '}'
        return out

    def generate_bias(self, layer: torch.nn.Linear):
        bias = layer.bias.data.cpu().detach().numpy()
        out = '{'
        for i in range(bias.shape[0]):
            out += f"{bias[i]:.7f}, "
        out += '}'
        return out

    def generate_function(self, function_name):
        out = f"LargeVector<{self.n_features_out}> {function_name}(LargeVector<{self.n_features_in}> input)"
        out += ' {\n'

        for id, layer in enumerate(self._fc):
            out += f"    LinearLayer<{layer.in_features}, {layer.out_features}> layer_{id};\n"
            out += f"    layer_{id}.weight.value = {self.generate_weight(layer)};\n"
            out += f"    layer_{id}.bias.value = {self.generate_bias(layer)};\n"

        for id in range(self.n_hidden_layers + 1):
            if id == 0:
                last_out = 'input'
            else:
                last_out = f"hidden_{id-1}"

            if id == self.n_hidden_layers:
                this_out = 'output'    
            else:
                this_out = f"hidden_{id}"

            out += f"    var {this_out} = layer_{id}.forward({last_out});\n"

            if id != self.n_hidden_layers:
                out += f"    {this_out} = relu({this_out});\n"

        out += f"    return {this_out};\n"
        out += "}\n"
        
        return out
