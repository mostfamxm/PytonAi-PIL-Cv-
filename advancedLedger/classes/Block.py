from dataclasses import dataclass
import dataclasses as dc
from datetime import datetime
import TransactionException
import rsa
from utilities import get_fields_str
import hashlib
import Message
from typing import Callable, List


@dataclass
class Block:
    prev_block_hash: str
    index: int
    messages: List[Message]
    current_block_hash: str = dc.field(init=False)
    time_added: datetime = dc.field(init=False)
    nonce: int = dc.field(init=False)

    TOKEN_PRIZE = 3

    def compute_block_header(self) -> bytes:
        block_str = \
            get_fields_str(self.prev_block_hash, self.index, self.messages, self.time_added, self.nonce)
        return block_str.encode()

    def compute_block_hash(self) -> str:
        """
        computes the hash of the block using SHA-256 algorithm
        :return: the hash in hexadecimal
        """
        block_hash = hashlib.sha256(self.compute_block_header()).hexdigest()
        self.current_block_hash = block_hash
        return block_hash

    def validate_single_block(self, block_obj) -> bool:
        """
        checks if the block is validate by comparing hash
        :param block_obj: object of type block
        :return: True if the block is validate else returns false
        """
        if self.compute_block_hash() == block_obj.compute_block_hash():
            return True
        return False


