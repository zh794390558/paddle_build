
import torch

a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])
neg_a = -a
scalar_b = torch.tensor(2.0)

def test_float32(a, b):
    ret = torch.floor_divide(a,b)
    print("test_float32")
    print(a, b)
    print(ret)
    return ret

def test_float64(a, b):
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    ret = torch.floor_divide(a,b)
    print("test_float64")
    print(a, b)
    print(ret)
    return ret

def test_float16(a, b):
    a = a.to(torch.float16)
    b = b.to(torch.float16)
    ret = torch.floor_divide(a,b)
    print("test_float16")
    print(a, b)
    print(ret)
    return ret

def test_bfloat16(a, b):
    a = a.to(torch.bfloat16)
    b = b.to(torch.bfloat16)
    ret = torch.floor_divide(a,b)
    print("test_bfloat16")
    print(a, b)
    print(ret)
    return ret

def test_int64(a, b):
    a = a.to(torch.int64)
    b = b.to(torch.int64)
    ret = torch.floor_divide(a,b)
    print("test_int64")
    print(a, b)
    print(ret)
    return ret

def test_int32(a, b):
    a = a.to(torch.int32)
    b = b.to(torch.int32)
    ret = torch.floor_divide(a,b)
    print("test_int32")
    print(a, b)
    print(ret)
    return ret

def test_int16(a, b):
    a = a.to(torch.int16)
    b = b.to(torch.int16)
    ret = torch.floor_divide(a,b)
    print("test_int16")
    print(a, b)
    print(ret)
    return ret


def test_int8(a, b):
    a = a.to(torch.int8)
    b = b.to(torch.int8)
    ret = torch.floor_divide(a,b)
    print("test_int8")
    print(a, b)
    print(ret)
    return ret

def test_uint8(a, b):
    a = a.to(torch.uint8)
    b = b.to(torch.uint8)
    ret = torch.floor_divide(a,b)
    print("test_uint8")
    print(a, b)
    print(ret)
    return ret

def test_bool(a, b):
    a = a.to(torch.bool)
    b = b.to(torch.bool)
    ret = torch.floor_divide(a,b)
    print("test_bool")
    print(a, b)
    print(ret)
    return ret

if __name__ == "__main__":
    test_bfloat16(a, b)
    test_bfloat16(neg_a, b)
    test_bfloat16(neg_a, scalar_b)

    test_float16(a, b)
    test_float16(neg_a, b)

    test_float32(a, b)
    test_float32(neg_a, b)

    test_float64(a, b)
    test_float64(neg_a, b)

    print("-"*10)

    test_int16(a, b)
    test_int16(neg_a, b)

    test_int8(a, b)
    test_int8(neg_a, b)

    test_uint8(a, b)
    test_uint8(neg_a, b)

    try:
        test_bool(a, b)
    except Exception as e:
        print(e)