from typing import Optional

import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models

def reduce_and_concat(tensor: torch.Tensor) -> torch.Tensor:
    """
    768차원 벡터를 3차원씩 평균을 내어 256차원으로 줄인 후, 원래 벡터와 결합하여 1024차원 벡터를 만듭니다.

    Args:
        tensor (torch.Tensor): 원래의 768차원 벡터 (seq_len, 768)

    Returns:
        torch.Tensor: 결합된 1024차원 벡터 (seq_len, 1024)
    """
    seq_len = tensor.size(0)
    # (seq_len, 768)을 (seq_len, 256, 3)으로 변환
    reshaped_tensor = tensor.view(seq_len, 256, 3)
    # 3차원씩 평균을 내어 (seq_len, 256)으로 축소
    reduced_tensor = reshaped_tensor.mean(dim=-1)
    # 원래의 768차원 벡터와 결합하여 (seq_len, 1024)로 확장
    concatenated_tensor = torch.cat((tensor, reduced_tensor), dim=-1)
    return concatenated_tensor


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    영어 텍스트에서 BERT의 특징을 추출합니다.

    Args:
        text (str): 영어 텍스트
        word2ph (list[int]): 원본 텍스트의 각 문자에 할당된 음소의 개수를 나타내는 리스트
        device (str): 추론에 사용할 디바이스
        assist_text (Optional[str], optional): 보조 텍스트 (기본값: None)
        assist_text_weight (float, optional): 보조 텍스트의 가중치 (기본값: 0.7)

    Returns:
        torch.Tensor: BERT의 특징
    """

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = bert_models.load_model(Languages.EN_BASE).to(device)  # type: ignore

    style_res_mean = None
    with torch.no_grad():
        tokenizer = bert_models.load_tokenizer(Languages.EN_BASE)
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  # type: ignore
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        if assist_text:
            style_inputs = tokenizer(assist_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)  # type: ignore
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - assist_text_weight)
                + style_res_mean.repeat(word2phone[i], 1) * assist_text_weight
            )
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 차원 축소 및 결합
    phone_level_feature = reduce_and_concat(phone_level_feature)

    return phone_level_feature.T