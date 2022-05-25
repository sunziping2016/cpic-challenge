from transformers import (
    T5Tokenizer,
    T5Config,
    T5ForConditionalGeneration,
)


def init_baseline_model(model_args, model_name, data_args, special_tokens=[]):
    print("init {} model from {}...".format(model_name, model_args.model_name_or_path))
    if model_name == "T5":
        tokenizer = T5Tokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=True,
            max_length=1024,
            truncation=True,
            additional_special_tokens=special_tokens,
        )
        config = T5Config.from_pretrained(model_args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
        model.config.max_length = data_args.val_max_target_length
        model.resize_token_embeddings(len(tokenizer))
    else:
        raise NotImplementedError("You can implement {} by yourself".format(model_name.upper()))

    return tokenizer, model
