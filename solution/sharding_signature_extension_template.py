

# --- recovered distributed-sharding signature compatibility extension ---
# This extension is self-contained and intentionally does not depend on torch at call time.
_SHARDING_SIGNATURE_WORD = __WORD__
_SHARDING_SIGNATURE_S1 = __S1__
_SHARDING_SIGNATURE_IW = __IW__
_SHARDING_SIGNATURE_IO = __IO__
_SHARDING_SIGNATURE_S3 = __S3__
_SHARDING_SIGNATURE_M = __M__
_SHARDING_SIGNATURE_P = __P__
_SHARDING_SIGNATURE_TABLE = __TABLE__
_SHARDING_SIGNATURE_ROTS = __ROTS__
_SHARDING_SIGNATURE_WMASK = (1 << _SHARDING_SIGNATURE_WORD) - 1
_SHARDING_SIGNATURE_R1MASK = _SHARDING_SIGNATURE_WMASK ^ ((1 << _SHARDING_SIGNATURE_S1) - 1)
_SHARDING_SIGNATURE_R2MASK = _SHARDING_SIGNATURE_WMASK ^ ((1 << (_SHARDING_SIGNATURE_IO + _SHARDING_SIGNATURE_IW)) - 1)
_SHARDING_SIGNATURE_R3MASK = (1 << (_SHARDING_SIGNATURE_WORD - _SHARDING_SIGNATURE_S3)) - 1
_SHARDING_SIGNATURE_IMASK = (1 << _SHARDING_SIGNATURE_IW) - 1


def _sharding_signature_order(count, seed):
    order = list(range(count))
    if seed is None:
        return order
    state = seed & ((1 << 64) - 1)
    for index in range(count - 1, 0, -1):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        swap = (state >> 33) % (index + 1)
        order[index], order[swap] = order[swap], order[index]
    return order


def _sharding_signature_codes(query):
    count, world = query["N"], query["W"]
    order = _sharding_signature_order(count, query["seed"])
    positions = [0] * count
    for position, global_index in enumerate(order):
        positions[global_index] = position
    if query["tail"] == "drop_last":
        kept = (count // world) * world
        assignment_length = kept
    else:
        kept = count
        assignment_length = ((count + world - 1) // world) * world
    per_rank = assignment_length // world
    rank_codes = [101, 134, 167, 200, 233, 266, 299, 332]
    codes = []
    for global_index in query["indices"]:
        position = positions[global_index]
        if query["tail"] == "drop_last" and position >= kept:
            codes.append(68)
        elif query["mode"] == "contiguous":
            codes.append(rank_codes[position // per_rank])
        else:
            codes.append(rank_codes[position % world])
    return codes


def _sharding_signature_rotl(value, rotation):
    rotation %= _SHARDING_SIGNATURE_WORD
    return ((value << rotation) | (value >> (_SHARDING_SIGNATURE_WORD - rotation))) & _SHARDING_SIGNATURE_WMASK


def sharding_signature(query):
    """Return the recovered assignment signature for a distributed sharding query."""
    codes = _sharding_signature_codes(query)
    accumulator = 0
    for code in codes:
        accumulator = (accumulator * _SHARDING_SIGNATURE_P + code) % _SHARDING_SIGNATURE_M
    value = accumulator & _SHARDING_SIGNATURE_WMASK
    value ^= (value << (1 + (value & ((1 << _SHARDING_SIGNATURE_S1) - 1)))) & _SHARDING_SIGNATURE_R1MASK
    value ^= (_SHARDING_SIGNATURE_TABLE[(value >> _SHARDING_SIGNATURE_IO) & _SHARDING_SIGNATURE_IMASK]
              << (_SHARDING_SIGNATURE_IO + _SHARDING_SIGNATURE_IW)) & _SHARDING_SIGNATURE_R2MASK
    value ^= (value >> (1 + ((value >> (_SHARDING_SIGNATURE_WORD - _SHARDING_SIGNATURE_S3))
                             & ((1 << _SHARDING_SIGNATURE_S3) - 1)))) & _SHARDING_SIGNATURE_R3MASK
    return _sharding_signature_rotl(value & _SHARDING_SIGNATURE_WMASK,
                                    _SHARDING_SIGNATURE_ROTS[len(codes) % 5])
