import numpy as np
qh = np.frombuffer(b'\x00' * 64, dtype=np.uint8)
qh_expanded = np.column_stack([qh] * 4)
shifts = np.array([6, 4, 2, 0], dtype=np.uint32)
qh_high = ((qh_expanded >> shifts) & 3).astype(np.float32).ravel()
print('shape:', qh_high.shape)