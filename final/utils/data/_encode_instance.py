from typing import Dict, List, Tuple

from utils.config import load_config


def encode_instance(ins: Dict[str, str], tokenizer, max_len: int) -> Dict[str, List[int]]:
    """Encode instance
    
    [Params]
    ins     : Dict[str, str]
    tokenizer
    max_len : int
    
    [Return]
    {input_ids, labels_rationale, labels_output}
    """
    try:
        x, rationale, y = ins['input'], ins['cot_process'], ins['output']
    except:
        x, rationale, y = ins['input'], '', ins['output']
    
    
    config = load_config()
    rationale_labels = config['SPECIAL_TOKENS']['RATIONALE']
    output_labels = config['SPECIAL_TOKENS']['OUTPUT']
    
    t_r_open = rationale_labels[0]
    t_r_close = rationale_labels[1]
    t_o_open = output_labels[0]
    t_o_close = output_labels[1]
    
    # ── 컨텍스트(입력) ──────────────────────────────
    ids_x        = tokenizer(x,               add_special_tokens=False).input_ids
    ids_r_open   = tokenizer(t_r_open,        add_special_tokens=False).input_ids
    ids_r        = tokenizer(rationale or '', add_special_tokens=False).input_ids
    ids_r_close  = tokenizer(t_r_close,       add_special_tokens=False).input_ids
    ids_o_open   = tokenizer(t_o_open,        add_special_tokens=False).input_ids
    ids_o        = tokenizer(y,               add_special_tokens=False).input_ids
    ids_o_close  = tokenizer(t_o_close,       add_special_tokens=False).input_ids
    
    seq = (
        ids_x + ids_r_open + ids_r + ids_r_close +
        ids_o_open + ids_o + ids_o_close
    )[:max_len]

    # ── 레이블 ────────────────────────────────────
    lbl_r = [-100] * len(seq)
    lbl_o = [-100] * len(seq)

    # rationale 예측 영역
    r_beg = len(ids_x) + len(ids_r_open)
    r_end = min(r_beg + len(ids_r), len(seq))
    lbl_r[r_beg:r_end] = seq[r_beg:r_end]

    # output 예측 영역
    o_beg = r_end + len(ids_r_close) + len(ids_o_open)
    lbl_o[o_beg:] = seq[o_beg:]
    
    # print(decode_labels({
    #     "input_ids": seq,
    #     "labels_rationale": lbl_r,
    #     "labels_output": lbl_o,
    # }, tokenizer), flush=True); exit()

    return {
        "input_ids": seq,
        "labels_rationale": lbl_r,
        "labels_output": lbl_o,
    }

def decode_labels(encoded: Dict[str, List[int]], tokenizer) -> Dict[str, str]:
    """
    encode_instance가 만든 딕셔너리를 받아
    - 전체 시퀀스
    - rationale 레이블 구간
    - output   레이블 구간
    을 사람이 읽을 수 있는 문자열로 반환
    """
    ids  = encoded["input_ids"]
    labR = encoded["labels_rationale"]
    labO = encoded["labels_output"]

    # 토큰 ID → 텍스트
    full_text = tokenizer.decode(ids, skip_special_tokens=False)

    r_ids = [tid for tid, lbl in zip(ids, labR) if lbl != -100]
    o_ids = [tid for tid, lbl in zip(ids, labO) if lbl != -100]

    rationale_text = tokenizer.decode(r_ids, skip_special_tokens=False)
    output_text    = tokenizer.decode(o_ids, skip_special_tokens=False)

    return {
        "full":      full_text,
        "rationale": rationale_text,
        "output":    output_text,
    }