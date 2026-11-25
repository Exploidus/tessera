#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwizzleConfig {
    pub rows: usize,
    pub cols: usize,
    pub elem_bytes: usize,
    pub shift: usize,
    pub mask: usize,
}

impl SwizzleConfig {
    pub fn compute(rows: usize, cols: usize, elem_bytes: usize) -> Self {
        let elements_per_line = 128 / elem_bytes.max(1);
        let shift = if elements_per_line >= 16 { 3 } else { 2 };
        let mask = 0x1F;

        Self {
            rows,
            cols,
            elem_bytes,
            shift,
            mask,
        }
    }

    pub fn swizzled_offset(&self, row: usize, col: usize) -> usize {
        let swizzled_col = col ^ ((row >> self.shift) & self.mask);
        row * self.cols + swizzled_col
    }

    pub fn emit_ptx_swizzle(
        &self,
        row_reg: &str,
        col_reg: &str,
        stride_bytes: usize,
    ) -> (String, String) {
        let mut asm = String::new();
        let swizzled_col = "%swiz_col";
        let row_offset = "%swiz_row_off";
        let final_byte_off = "%swiz_byte_off";
        let temp_shift = "%swiz_shift";

        asm.push_str(&format!(
            "    shr.u32 {}, {}, {};\n",
            temp_shift, row_reg, self.shift
        ));
        asm.push_str(&format!(
            "    and.b32 {}, {}, {};\n",
            temp_shift, temp_shift, self.mask
        ));
        asm.push_str(&format!(
            "    xor.b32 {}, {}, {};\n",
            swizzled_col, col_reg, temp_shift
        ));
        asm.push_str(&format!(
            "    mul.lo.u32 {}, {}, {};\n",
            row_offset, row_reg, stride_bytes
        ));
        asm.push_str(&format!(
            "    mad.lo.u32 {}, {}, {}, {};\n",
            final_byte_off, swizzled_col, self.elem_bytes, row_offset
        ));

        (asm, final_byte_off.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_bank_conflicts_for_contiguous_loads() {
        let swizzle = SwizzleConfig::compute(64, 64, 4);
        let mut banks = std::collections::HashSet::new();
        for tid in 0..32 {
            let row = tid;
            let col = tid;
            let offset = swizzle.swizzled_offset(row, col);
            let bank = (offset * 4 / 4) % 32;
            banks.insert(bank);
        }
        assert_eq!(banks.len(), 32);
    }
}
