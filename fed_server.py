
def aggregate(model_target, model_list, n_sample_list):
    n_sample_in_total = 0
    for model_source, n_sample in zip(model_list, n_sample_list):
        n_sample_in_total += n_sample
        shared_list, model = model_source.upload_model()
        for md_target, md_source in zip(model_target.model.named_parameters(), model.named_parameters()):
            if md_source[0].split(".")[0] in shared_list:
                md_target[1].data += n_sample * md_source[1].clone().detach().data
    for md_target in model_target.model.named_parameters():
        if md_target[0].split(".")[0] in model_target.shared_list:
            md_target[1].data = md_target[1].clone().detach().data / n_sample_in_total
    return model_target
