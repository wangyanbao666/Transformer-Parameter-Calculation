from transformers import T5ForConditionalGeneration
from prettytable import PrettyTable


def print_model_params(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    count_parameters(model)
    print("model: ", model)
    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# print_model_params("google-t5/t5-3b")
# print_model_params("google-t5/t5-large")
# print_model_params("google-t5/t5-base")
print_model_params("google/flan-t5-base")