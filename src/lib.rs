/**
Each stack frame has its own set of 256 registers.

<u64> = uint64_t.
<Reg> = uint8_t.
<string> = <length:u32> <char>...

<Program> = <Instr>...

<Instr> = fd <Name:string> <nargs:u64> <nskip:n64>  # function definition
        | mov <Reg> <Val>
        | ineg <Reg>
        | iadd <Reg> <Val>
        | isub <Reg> <Val>
        | add <Reg> <Val>
        | sub <Reg> <Val>
        | umul <Reg> <Val>
        | smul <Reg> <Val>
        | urem <Reg> <Val>
        | srem <Reg> <Val>
        | udiv <Reg> <Val>
        | sdiv <Reg> <Val>
        | and <Reg> <Val>
        | or <Reg> <Val>
        | xor <Reg> <Val>
        | shl <Reg> <Val>
        | lshr <Reg> <Val>
        | ashr <Reg> <Val>
        | fadd <Reg> <Val>
        | fsub <Reg> <Val>
        | fmul <Reg> <Val>
        | frem <Reg> <Val>
        | fdiv <Reg> <Val>
        | eq <Reg> <Val>           # == bitwise
        | ne <Reg> <Val>           # != bitwise
        | slt <Reg> <Val>          # <  as signed integers
        | sle <Reg> <Val>          # <= as signed integers
        | sgt <Reg> <Val>          # >  as signed integers
        | sge <Reg> <Val>          # >= as signed integers
        | ult <Reg> <Val>          # <  as unsigned integers
        | ule <Reg> <Val>          # <= as unsigned integers
        | ugt <Reg> <Val>          # >  as unsigned integers
        | uge <Reg> <Val>          # >= as unsigned integers
        | ld8 <Reg> <Val>          # <Reg> = *(uint8_t *)<Val>
        | ld16 <Reg> <Val>         # <Reg> = *(uint16_t *)<Val>
        | ld32 <Reg> <Val>         # <Reg> = *(uint32_t *)<Val>
        | ld64 <Reg> <Val>         # <Reg> = *(uint64_t *)<Val>
        | st8 <Val> <Reg>          # *(uint8_t *)<Val> = <Reg>
        | st16 <Val> <Reg>         # *(uint16_t *)<Val> = <Reg>
        | st32 <Val> <Reg>         # *(uint32_t *)<Val> = <Reg>
        | st64 <Val> <Reg>         # *(uint64_t *)<Val> = <Reg>
        | jmp <Off>                # unconditional relative jump
        | jz <Reg> <Off>           # relative jump if <Reg> == 0
        | lea <Reg1> <Reg2>        # <Reg1> = &<Reg2> (load effective address)
        | leave                    # leave function returning nothing
        | ret <Val>                # return <Val> from function
        | gg <Name:string> <Reg>   # get global
        | sg <Name:string> <Reg>   # set global
        | css <string> <Reg>       # create static string; write pointer to <Reg>
        | css_dyn <Reg1> <Reg2>    # create static string of length 8 and put <Reg2> into it
        | call0 <Reg>
        | call1 <Reg> <Reg1>
        | call2 <Reg> <Reg1> <Reg2>
        | call3 <Reg> <Reg1> <Reg2> <Reg3>
        | call4 <Reg> <Reg1> <Reg2> <Reg3> <Reg4>
        | call5 <Reg> <Reg1> <Reg2> <Reg3> <Reg4> <Reg5>
        | call6 <Reg> <Reg1> <Reg2> <Reg3> <Reg4> <Reg5> <Reg6>
        | call7 <Reg> <Reg1> <Reg2> <Reg3> <Reg4> <Reg5> <Reg6> <Reg7>
        | call8 <Reg> <Reg1> <Reg2> <Reg3> <Reg4> <Reg5> <Reg6> <Reg7> <Reg8>

call<N> functions put the return value into the function register (<Reg>).

Instructions that may take either a register or constant operand (<Val>) are encoded as follows:
    <instruction byte> <byte with value 0> <Reg>
or
    <instruction byte> <byte with value 1> <Constant:u64>
*/
#[macro_use]
extern crate failure;
#[macro_use]
extern crate runtime_fmt;
extern crate libvm;
use crate::allocator::Allocator;
use crate::error::RuntimeError;
use crate::instruction::{Program, RevmInstruction, Value};
use crate::libvm::Cpu;
use crate::memory::Memory;
use crate::register_set::RegisterSet;
use failure::Error;
//use libc::scanf;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::EPSILON;
use std::iter::Iterator;
use std::ops::Rem;
use std::rc::Rc;

mod allocator;
mod error;
pub mod instruction;
mod memory;
mod register_set;

type CpuFn = Box<dyn Fn(&NativeFunctions, Vec<u8>) -> Result<u64, Error>>;
#[repr(C)]
enum Function {
    Native(CpuFn),
    UserDefined(usize, u64, u64),
}

struct NativeFunctions {
    allocator: Rc<RefCell<Allocator>>,
    register_stack: Rc<RefCell<Vec<RegisterSet>>>,
}

impl NativeFunctions {
    fn puts(&self, args: Vec<u8>) -> Result<u64, Error> {
        if args.len() != 1 {
            return Err(RuntimeError::WrongArgumentsNumber {
                name: "puts".to_owned(),
                expected: 1,
                got: args.len(),
            }
            .into());
        }
        self.register_stack
            .borrow()
            .last()
            .unwrap()
            .to_string(args[0] as usize)
            .map(|s| {
                println!("{}", s);
                0
            })
    }

    fn printf(&self, args: Vec<u8>) -> Result<u64, Error> {
        let rs = self.register_stack.borrow();
        let registers: &RegisterSet = rs.last().unwrap();
        let r = match args.len() {
            1 => {
                let content = registers.to_string(args[0] as usize)?;
                rt_println!(content).unwrap();
                Ok(0)
            }
            2 => {
                let content = registers.to_string(args[0] as usize)?;
                rt_println!(content, registers.to_string(args[1] as usize)?).unwrap();
                Ok(0)
            }
            3 => {
                let content = registers.to_string(args[0] as usize)?;
                rt_println!(
                    content,
                    registers.to_string(args[1] as usize)?,
                    registers.to_string(args[2] as usize)?,
                )
                .unwrap();
                Ok(0)
            }
            4 => {
                let content = registers.to_string(args[0] as usize)?;
                rt_println!(
                    content,
                    registers.to_string(args[1] as usize)?,
                    registers.to_string(args[2] as usize)?,
                    registers.to_string(args[3] as usize)?,
                )
                .unwrap();
                Ok(0)
            }
            5 => {
                let content = registers.to_string(args[0] as usize)?;
                rt_println!(
                    content,
                    registers.to_string(args[1] as usize)?,
                    registers.to_string(args[2] as usize)?,
                    registers.to_string(args[3] as usize)?,
                    registers.to_string(args[4] as usize)?,
                )
                .unwrap();
                Ok(0)
            }
            6 => {
                let content = registers.to_string(args[0] as usize)?;
                rt_println!(
                    content,
                    registers.to_string(args[1] as usize)?,
                    registers.to_string(args[2] as usize)?,
                    registers.to_string(args[3] as usize)?,
                    registers.to_string(args[4] as usize)?,
                    registers.to_string(args[5] as usize)?,
                )
                .unwrap();
                Ok(0)
            }
            7 => {
                let content = registers.to_string(args[0] as usize)?;
                rt_println!(
                    content,
                    registers.to_string(args[1] as usize)?,
                    registers.to_string(args[2] as usize)?,
                    registers.to_string(args[3] as usize)?,
                    registers.to_string(args[4] as usize)?,
                    registers.to_string(args[5] as usize)?,
                    registers.to_string(args[6] as usize)?,
                )
                .unwrap();
                Ok(0)
            }
            8 => {
                let content = registers.to_string(args[0] as usize)?;
                rt_println!(
                    content,
                    registers.to_string(args[1] as usize)?,
                    registers.to_string(args[2] as usize)?,
                    registers.to_string(args[3] as usize)?,
                    registers.to_string(args[4] as usize)?,
                    registers.to_string(args[5] as usize)?,
                    registers.to_string(args[6] as usize)?,
                    registers.to_string(args[7] as usize)?,
                )
                .unwrap();
                Ok(0)
            }
            n => Err(RuntimeError::WrongArgumentsNumber {
                name: "printf".to_owned(),
                expected: 8,
                got: n,
            }),
        }?;
        Ok(r)
    }

    /*
    fn scanf(&self, args: Vec<u8>) -> Result<u64, Error> {
        if args.is_empty() {
            Err(RuntimeError::WrongArgumentsNumber {
                name: "scanf".to_owned(),
                expected: 8,
                got: 0,
            })?;
        }
        let rc = self.register_stack.borrow();
        let registers = rc.last().unwrap();
        let content: Vec<i8> = registers
            .to_string(args[0] as usize)?
            .into_bytes()
            .iter()
            .map(|v| *v as i8)
            .collect();
        let mut args_ptr = args.clone();
        let r = match args.len() {
            1 => Ok(unsafe { scanf((&content).as_ptr()) }),
            2 => Ok(unsafe { scanf((&content).as_ptr(), args_ptr.as_mut_ptr().add(1)) }),
            3 => Ok(unsafe {
                scanf(
                    (&content).as_ptr(),
                    args_ptr.as_mut_ptr().add(1),
                    args_ptr.as_mut_ptr().add(2),
                )
            }),
            4 => Ok(unsafe {
                scanf(
                    (&content).as_ptr(),
                    args_ptr.as_mut_ptr().add(1),
                    args_ptr.as_mut_ptr().add(2),
                    args_ptr.as_mut_ptr().add(3),
                )
            }),
            5 => Ok(unsafe {
                scanf(
                    (&content).as_ptr(),
                    args_ptr.as_mut_ptr().add(1),
                    args_ptr.as_mut_ptr().add(2),
                    args_ptr.as_mut_ptr().add(3),
                    args_ptr.as_mut_ptr().add(4),
                )
            }),
            6 => Ok(unsafe {
                scanf(
                    (&content).as_ptr(),
                    args_ptr.as_mut_ptr().add(1),
                    args_ptr.as_mut_ptr().add(2),
                    args_ptr.as_mut_ptr().add(3),
                    args_ptr.as_mut_ptr().add(4),
                    args_ptr.as_mut_ptr().add(5),
                )
            }),
            7 => Ok(unsafe {
                scanf(
                    (&content).as_ptr(),
                    args_ptr.as_mut_ptr().add(1),
                    args_ptr.as_mut_ptr().add(2),
                    args_ptr.as_mut_ptr().add(3),
                    args_ptr.as_mut_ptr().add(4),
                    args_ptr.as_mut_ptr().add(5),
                    args_ptr.as_mut_ptr().add(6),
                )
            }),
            8 => Ok(unsafe {
                scanf(
                    (&content).as_ptr(),
                    args_ptr.as_mut_ptr().add(1),
                    args_ptr.as_mut_ptr().add(2),
                    args_ptr.as_mut_ptr().add(3),
                    args_ptr.as_mut_ptr().add(4),
                    args_ptr.as_mut_ptr().add(5),
                    args_ptr.as_mut_ptr().add(6),
                    args_ptr.as_mut_ptr().add(7),
                )
            }),
            n => Err(RuntimeError::WrongArgumentsNumber {
                name: "scanf".to_owned(),
                expected: 8,
                got: n,
            }),
        }?;
        Ok(r as u64)
    }
    */

    fn exit(&self, args: Vec<u8>) -> Result<u64, Error> {
        if args.len() != 1 {
            return Err(RuntimeError::WrongArgumentsNumber {
                name: "exit".to_owned(),
                expected: 1,
                got: args.len(),
            }
            .into());
        }
        Err(RuntimeError::ProgramEnded {
            errno: self.register_stack.borrow().last().unwrap().get(0)?,
        })?;
        Ok(0)
    }

    fn malloc(&self, args: Vec<u8>) -> Result<u64, Error> {
        if args.len() != 1 {
            return Err(RuntimeError::WrongArgumentsNumber {
                name: "malloc".to_owned(),
                expected: 1,
                got: args.len(),
            }
            .into());
        }
        let size = self.register_stack.borrow().last().unwrap().get(0)? as usize;
        self.allocator.borrow_mut().malloc(size).map(|v| v as u64)
    }

    fn free(&self, args: Vec<u8>) -> Result<u64, Error> {
        if args.len() != 1 {
            return Err(RuntimeError::WrongArgumentsNumber {
                name: "free".to_owned(),
                expected: 1,
                got: args.len(),
            }
            .into());
        }
        let address = self.register_stack.borrow().last().unwrap().get(0)? as usize;
        self.allocator.borrow_mut().free(address).map(|_| 0)
    }
}

pub struct CpuRevm {
    allocator: Rc<RefCell<Allocator>>,
    call_stack: Vec<(usize, u8)>,
    pub(crate) functions: HashMap<String, Function>,
    globals: HashMap<String, u64>,
    memory: Memory,
    pc: usize,
    register_stack: Rc<RefCell<Vec<RegisterSet>>>,
    program: Program,
}

impl Cpu<RevmInstruction> for CpuRevm {
    fn can_run(&self) -> bool {
        true
    }

    fn is_done(&self) -> bool {
        !self.program.0.is_empty()
    }

    fn get_next_instruction(&mut self) -> Option<RevmInstruction> {
        self.program.0.pop()
    }

    fn increase_pc(&mut self, steps: usize) {
        self.pc += steps;
    }

    fn get_pc(&self) -> usize {
        self.pc
    }

    fn execute_instruction(&mut self, instruction: RevmInstruction) -> Result<(), Error> {
        match instruction {
            RevmInstruction::Fd { name, args, skip } => {
                self.functions
                    .insert(name.clone(), Function::UserDefined(self.pc, args, skip));
                self.pc += skip as usize;
            }
            RevmInstruction::Mov { register, value } => self.value_to_register(register, value)?,
            RevmInstruction::Gg { string, register } => {
                let value = *self
                    .globals
                    .get(&string)
                    .ok_or(RuntimeError::GlobalNotFound {
                        name: string.clone(),
                    })?;
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                registers.set(register as usize, value)?;
            }
            RevmInstruction::Sg { string, register } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let value = registers.get(register as usize)?;
                self.globals.insert(string, value);
            }
            RevmInstruction::Css { string, register } => {
                let string_size = (string.len() as f64 / 8f64).ceil() as usize;
                let address = self.allocator.borrow_mut().malloc(string_size)?;
                self.memory.copy_u8_vector(string.as_bytes(), address);
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                registers.set(register as usize, address as u64)?;
            }
            RevmInstruction::Ld8 { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                registers.set(
                    register as usize,
                    match value {
                        Value::Register(source) => u64::from(registers.get(source as usize)? as u8),
                        Value::Constant(value) => value,
                    },
                )?;
            }
            RevmInstruction::Ld16 { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                registers.set(
                    register as usize,
                    match value {
                        Value::Register(source) => {
                            u64::from(registers.get(source as usize)? as u16)
                        }
                        Value::Constant(value) => value,
                    },
                )?;
            }
            RevmInstruction::Ld32 { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                registers.set(
                    register as usize,
                    match value {
                        Value::Register(source) => {
                            u64::from(registers.get(source as usize)? as u32)
                        }
                        Value::Constant(value) => value,
                    },
                )?;
            }
            RevmInstruction::Ld64 { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                registers.set(
                    register as usize,
                    match value {
                        Value::Register(source) => registers.get(source as usize)?,
                        Value::Constant(value) => value,
                    },
                )?;
            }
            RevmInstruction::St8 {
                register,
                value: address_value,
            } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let value = registers.get(register as usize)? as u8;
                let address = match address_value {
                    Value::Constant(a) => a as usize,
                    Value::Register(r) => registers.get(r as usize)? as usize,
                };
                self.memory.copy_u8(value, address);
            }
            RevmInstruction::St16 {
                register,
                value: address_value,
            } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let value = registers.get(register as usize)? as u16;
                let address = match address_value {
                    Value::Constant(a) => a as usize,
                    Value::Register(r) => registers.get(r as usize)? as usize,
                };
                self.memory.copy_u16(value, address);
            }
            RevmInstruction::St32 {
                register,
                value: address_value,
            } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let value = registers.get(register as usize)? as u32;
                let address = match address_value {
                    Value::Constant(a) => a as usize,
                    Value::Register(r) => registers.get(r as usize)? as usize,
                };
                self.memory.copy_u32(value, address);
            }
            RevmInstruction::St64 {
                register,
                value: address_value,
            } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let value = registers.get(register as usize)? as u64;
                let address = match address_value {
                    Value::Constant(a) => a as usize,
                    Value::Register(r) => registers.get(r as usize)? as usize,
                };
                self.memory.copy_u64(value, address);
            }
            RevmInstruction::Lea { destiny, source } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let effective_address = registers.address + source as usize;
                registers.set(destiny as usize, effective_address as u64)?;
            }
            RevmInstruction::Iadd { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get_i64(register as usize)?;
                let new_value = destiny_value.wrapping_add(match value {
                    Value::Register(s) => registers.get_i64(s as usize)?,
                    Value::Constant(v) => v as i64,
                });
                registers.set_i64(register as usize, new_value);
            }
            RevmInstruction::Isub { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get_i64(register as usize)?;
                let new_value = destiny_value.wrapping_sub(match value {
                    Value::Register(s) => registers.get_i64(s as usize)?,
                    Value::Constant(v) => v as i64,
                });
                registers.set_i64(register as usize, new_value);
            }
            RevmInstruction::Smul { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get_i64(register as usize)?;
                let new_value = destiny_value.wrapping_mul(match value {
                    Value::Register(s) => registers.get_i64(s as usize)?,
                    Value::Constant(v) => v as i64,
                });
                registers.set_i64(register as usize, new_value);
            }
            RevmInstruction::Umul { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get(register as usize)?;
                let new_value = destiny_value.wrapping_mul(match value {
                    Value::Register(s) => registers.get(s as usize)?,
                    Value::Constant(v) => v,
                });
                registers.set(register as usize, new_value)?;
            }
            RevmInstruction::Srem { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get_i64(register as usize)?;
                let new_value = destiny_value.wrapping_rem(match value {
                    Value::Register(s) => registers.get_i64(s as usize)?,
                    Value::Constant(v) => v as i64,
                });
                registers.set_i64(register as usize, new_value);
            }
            RevmInstruction::Urem { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get(register as usize)?;
                let new_value = destiny_value.wrapping_rem(match value {
                    Value::Register(s) => registers.get(s as usize)?,
                    Value::Constant(v) => v,
                });
                registers.set(register as usize, new_value)?;
            }
            RevmInstruction::Sdiv { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get_i64(register as usize)?;
                let new_value = destiny_value.wrapping_div(match value {
                    Value::Register(s) => registers.get_i64(s as usize)?,
                    Value::Constant(v) => v as i64,
                });
                registers.set_i64(register as usize, new_value);
            }
            RevmInstruction::Udiv { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get(register as usize)?;
                let new_value = destiny_value.wrapping_div(match value {
                    Value::Register(s) => registers.get(s as usize)?,
                    Value::Constant(v) => v,
                });
                registers.set(register as usize, new_value)?;
            }
            RevmInstruction::And { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get(register as usize)?;
                let new_value = destiny_value
                    & (match value {
                    Value::Register(s) => registers.get(s as usize)?,
                    Value::Constant(v) => v,
                });
                registers.set(register as usize, new_value)?;
            }
            RevmInstruction::Or { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get(register as usize)?;
                let new_value = destiny_value
                    | (match value {
                    Value::Register(s) => registers.get(s as usize)?,
                    Value::Constant(v) => v,
                });
                registers.set(register as usize, new_value)?;
            }
            RevmInstruction::Xor { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get(register as usize)?;
                let new_value = destiny_value
                    ^ (match value {
                    Value::Register(s) => registers.get(s as usize)?,
                    Value::Constant(v) => v,
                });
                registers.set(register as usize, new_value)?;
            }
            RevmInstruction::Shl { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get(register as usize)?;
                let new_value = destiny_value.wrapping_shl(match value {
                    Value::Register(s) => registers.get(s as usize)?,
                    Value::Constant(v) => v,
                } as u32);
                registers.set(register as usize, new_value)?;
            }
            RevmInstruction::Ashr { register, value } => {
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                let destiny_value = registers.get(register as usize)?;
                let new_value = destiny_value.wrapping_shr(match value {
                    Value::Register(s) => registers.get(s as usize)?,
                    Value::Constant(v) => v,
                } as u32);
                registers.set(register as usize, new_value)?;
            }
            RevmInstruction::Lshr { register, value } => {
                self.lshr(register, value)?;
            }
            RevmInstruction::Ineg { register } => {
                self.ineg(register)?;
            }
            RevmInstruction::Fadd { register, value } => {
                self.fadd(register, value)?;
            }
            RevmInstruction::Fsub { register, value } => {
                self.fsub(register, value)?;
            }
            RevmInstruction::Fmul { register, value } => {
                self.fmul(register, value)?;
            }
            RevmInstruction::Frem { register, value } => {
                self.frem(register, value)?;
            }
            RevmInstruction::Fdiv { register, value } => {
                self.fdiv(register, value)?;
            }
            RevmInstruction::Eq { register, value } => {
                self.eq(register, value)?;
            }
            RevmInstruction::Ne { register, value } => {
                self.ne(register, value)?;
            }
            RevmInstruction::Ult { register, value } => {
                self.ult(register, value)?;
            }
            RevmInstruction::Ule { register, value } => {
                self.ule(register, value)?;
            }
            RevmInstruction::Ugt { register, value } => {
                self.ugt(register, value)?;
            }
            RevmInstruction::Uge { register, value } => {
                self.uge(register, value)?;
            }
            RevmInstruction::Slt { register, value } => {
                self.slt(register, value)?;
            }
            RevmInstruction::Sle { register, value } => {
                self.sle(register, value)?;
            }
            RevmInstruction::Sgt { register, value } => {
                self.sgt(register, value)?;
            }
            RevmInstruction::Sge { register, value } => {
                self.sge(register, value)?;
            }
            RevmInstruction::Feq { register, value } => {
                self.feq(register, value)?;
            }
            RevmInstruction::Fne { register, value } => {
                self.fne(register, value)?;
            }
            RevmInstruction::Flt { register, value } => {
                self.flt(register, value)?;
            }
            RevmInstruction::Fle { register, value } => {
                self.fle(register, value)?;
            }
            RevmInstruction::Fgt { register, value } => {
                self.fgt(register, value)?;
            }
            RevmInstruction::Fge { register, value } => {
                self.fge(register, value)?;
            }
            RevmInstruction::Jmp { offset } => {
                self.pc = ((self.pc as i64) + offset - 1) as usize;
            }
            RevmInstruction::Jnz { offset, register } => {
                self.jnz(offset, register)?;
            }
            RevmInstruction::Jz { offset, register } => {
                self.jz(offset, register)?;
            }
            RevmInstruction::Call {
                return_register,
                arguments,
            } => {
                self.call_function(return_register, arguments)?;
            }
            RevmInstruction::Ret { value } => {
                self.return_from_function(value)?;
            }
            RevmInstruction::Leave => {
                self.leave()?;
            }
            _ => panic!("Not implemented yet"),
        };
        Ok(())
    }
}

impl CpuRevm {
    pub fn new(capacity: usize) -> Result<CpuRevm, Error> {
        let memory = Memory::new(capacity);
        let allocator = Rc::new(RefCell::new(Allocator::new(capacity)));
        let register_stack = Rc::new(RefCell::new(vec![]));
        let mut functions = HashMap::new();
        functions.insert(
            "puts".to_owned(),
            Function::Native(Box::new(NativeFunctions::puts)),
        );
        /*
        TODO: Make scanf use the sandbox
        functions.insert(
            "scanf".to_owned(),
            Function::Native(Box::new(|cpu, args| cpu.scanf(args))),
        );
        */
        functions.insert(
            "printf".to_owned(),
            Function::Native(Box::new(NativeFunctions::printf)),
        );
        functions.insert(
            "exit".to_owned(),
            Function::Native(Box::new(NativeFunctions::exit)),
        );
        functions.insert(
            "malloc".to_owned(),
            Function::Native(Box::new(NativeFunctions::malloc)),
        );
        functions.insert(
            "free".to_owned(),
            Function::Native(Box::new(NativeFunctions::free)),
        );
        let mut cpu = CpuRevm {
            allocator,
            functions,
            memory,
            pc: 0,
            register_stack,
            call_stack: Vec::new(),
            globals: HashMap::new(),
            program: Program(vec![]),
        };
        cpu.add_function("puts")?;
        cpu.add_function("printf")?;
        cpu.add_function("malloc")?;
        cpu.add_function("exit")?;
        cpu.add_function("free")?;
        let register_set = cpu.create_new_register_set()?;
        cpu.register_stack.borrow_mut().push(register_set);
        Ok(cpu)
    }

    fn add_function(&mut self, name: &'static str) -> Result<(), Error> {
        match self.functions.get(name) {
            Some(f) => {
                let address = self.allocator.borrow_mut().malloc_t::<Function>()?;
                self.memory.copy_t(f, address);
            }
            None => Err(RuntimeError::GlobalNotFound {
                name: name.to_owned(),
            })?,
        };
        Ok(())
    }

    pub fn execute_program(&mut self, program: Program) -> Result<(), Error> {
        self.program = program;
        self.pc = 0;
        while !self.is_done() {
            let instruction = self
                .get_next_instruction()
                .ok_or(RuntimeError::NoMoreInstructions)?;
            self.execute_instruction(instruction)?;
            self.increase_pc(1);
        }
        Ok(())
    }

    fn lshr(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_u32(register as usize)?;
        let new_value = destiny_value.wrapping_shr(match value {
            Value::Register(s) => registers.get(s as usize)?,
            Value::Constant(v) => v,
        } as u32);
        registers.set_u32(register as usize, new_value);
        Ok(())
    }

    fn ineg(&mut self, register: u8) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        registers.set(register as usize, !registers.get(register as usize)?)?;
        Ok(())
    }

    fn fadd(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = destiny_value
            + (match value {
                Value::Register(s) => registers.get_f64(s as usize)?,
                Value::Constant(v) => v as f64,
            });
        registers.set_f64(register as usize, new_value);
        Ok(())
    }

    fn fsub(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = destiny_value
            - (match value {
                Value::Register(s) => registers.get_f64(s as usize)?,
                Value::Constant(v) => v as f64,
            });
        registers.set_f64(register as usize, new_value);
        Ok(())
    }

    fn fmul(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = destiny_value
            * (match value {
                Value::Register(s) => registers.get_f64(s as usize)?,
                Value::Constant(v) => v as f64,
            });
        registers.set_f64(register as usize, new_value);
        Ok(())
    }

    fn frem(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = destiny_value.rem(match value {
            Value::Register(s) => registers.get_f64(s as usize)?,
            Value::Constant(v) => v as f64,
        });
        registers.set_f64(register as usize, new_value);
        Ok(())
    }

    fn fdiv(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = destiny_value
            / (match value {
                Value::Register(s) => registers.get_f64(s as usize)?,
                Value::Constant(v) => v as f64,
            });
        registers.set_f64(register as usize, new_value);
        Ok(())
    }

    fn eq(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get(register as usize)?;
        let new_value = destiny_value
            == (match value {
                Value::Register(s) => registers.get(s as usize)?,
                Value::Constant(v) => v,
            });
        registers.set(register as usize, new_value as u64)?;
        Ok(())
    }

    fn ne(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get(register as usize)?;
        let new_value = destiny_value
            != (match value {
                Value::Register(s) => registers.get(s as usize)?,
                Value::Constant(v) => v,
            });
        registers.set(register as usize, new_value as u64)?;
        Ok(())
    }

    fn ult(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get(register as usize)?;
        let new_value = destiny_value
            < (match value {
                Value::Register(s) => registers.get(s as usize)?,
                Value::Constant(v) => v,
            });
        registers.set(register as usize, new_value as u64)?;
        Ok(())
    }

    fn ule(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get(register as usize)?;
        let new_value = destiny_value
            <= (match value {
                Value::Register(s) => registers.get(s as usize)?,
                Value::Constant(v) => v,
            });
        registers.set(register as usize, new_value as u64)?;
        Ok(())
    }

    fn ugt(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get(register as usize)?;
        let new_value = destiny_value
            > (match value {
                Value::Register(s) => registers.get(s as usize)?,
                Value::Constant(v) => v,
            });
        registers.set(register as usize, new_value as u64)?;
        Ok(())
    }

    fn uge(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get(register as usize)?;
        let new_value = destiny_value
            >= (match value {
                Value::Register(s) => registers.get(s as usize)?,
                Value::Constant(v) => v,
            });
        registers.set(register as usize, new_value as u64)?;
        Ok(())
    }

    fn slt(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_i64(register as usize)?;
        let new_value = destiny_value
            < (match value {
                Value::Register(s) => registers.get_i64(s as usize)?,
                Value::Constant(v) => v as i64,
            });
        registers.set_i64(register as usize, new_value as i64);
        Ok(())
    }

    fn sle(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_i64(register as usize)?;
        let new_value = destiny_value
            <= (match value {
                Value::Register(s) => registers.get_i64(s as usize)?,
                Value::Constant(v) => v as i64,
            });
        registers.set_i64(register as usize, new_value as i64);
        Ok(())
    }

    fn sgt(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_i64(register as usize)?;
        let new_value = destiny_value
            > (match value {
                Value::Register(s) => registers.get_i64(s as usize)?,
                Value::Constant(v) => v as i64,
            });
        registers.set_i64(register as usize, new_value as i64);
        Ok(())
    }

    fn sge(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_i64(register as usize)?;
        let new_value = destiny_value
            >= (match value {
                Value::Register(s) => registers.get_i64(s as usize)?,
                Value::Constant(v) => v as i64,
            });
        registers.set_i64(register as usize, new_value as i64);
        Ok(())
    }

    fn feq(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = (destiny_value
            - (match value {
                Value::Register(s) => registers.get_f64(s as usize)?,
                Value::Constant(v) => v as f64,
            }))
        .abs()
            < EPSILON;
        registers.set_i64(register as usize, new_value as i64);
        Ok(())
    }

    fn fne(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = (destiny_value
            - (match value {
                Value::Register(s) => registers.get_f64(s as usize)?,
                Value::Constant(v) => v as f64,
            }))
        .abs()
            >= EPSILON;
        registers.set_i64(register as usize, new_value as i64);
        Ok(())
    }

    fn flt(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = destiny_value
            < (match value {
                Value::Register(s) => registers.get_f64(s as usize)?,
                Value::Constant(v) => v as f64,
            });
        registers.set_f64(register as usize, new_value as i64 as f64);
        Ok(())
    }

    fn fle(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = destiny_value
            <= (match value {
                Value::Register(s) => registers.get_f64(s as usize)?,
                Value::Constant(v) => v as f64,
            });
        registers.set_f64(register as usize, new_value as i64 as f64);
        Ok(())
    }

    fn fgt(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = destiny_value
            > (match value {
                Value::Register(s) => registers.get_f64(s as usize)?,
                Value::Constant(v) => v as f64,
            });
        registers.set_f64(register as usize, new_value as i64 as f64);
        Ok(())
    }

    fn fge(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let destiny_value = registers.get_f64(register as usize)?;
        let new_value = destiny_value
            >= (match value {
                Value::Register(s) => registers.get_f64(s as usize)?,
                Value::Constant(v) => v as f64,
            });
        registers.set_f64(register as usize, new_value as i64 as f64);
        Ok(())
    }

    fn jnz(&mut self, offset: i64, register: u8) -> Result<(), Error> {
        let rc = self.register_stack.borrow();
        let registers = rc.last().unwrap();
        if registers.get(register as usize)? != 0 {
            self.pc = ((self.pc as i64) + offset - 1) as usize;
        }
        Ok(())
    }

    fn jz(&mut self, offset: i64, register: u8) -> Result<(), Error> {
        let rc = self.register_stack.borrow();
        let registers = rc.last().unwrap();
        if registers.get(register as usize)? == 0 {
            self.pc = ((self.pc as i64) + offset - 1) as usize;
        }
        Ok(())
    }

    fn leave(&mut self) -> Result<(), Error> {
        self.register_stack.borrow_mut().pop();
        let (new_pc, _) = self
            .call_stack
            .pop()
            .ok_or(RuntimeError::ReturnOnNoFunction)?;
        self.pc = new_pc;
        Ok(())
    }

    fn return_from_function(&mut self, value: Value) -> Result<(), Error> {
        let (new_i, r) = self
            .call_stack
            .pop()
            .ok_or(RuntimeError::ReturnOnNoFunction)?;
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        let ret_value = match value {
            Value::Constant(v) => v,
            Value::Register(r) => registers.get(r as usize)?,
        };
        self.register_stack.borrow_mut().pop();
        registers.set(r as usize, ret_value)?;
        self.pc = new_i;
        Ok(())
    }

    fn call_function(
        &mut self,
        return_register: u8,
        arguments: [Option<u8>; 8],
    ) -> Result<(), Error> {
        let native_functions = NativeFunctions {
            allocator: self.allocator.clone(),
            register_stack: self.register_stack.clone(),
        };
        let rc = self.register_stack.borrow();
        let function = {
            let registers = rc.last().unwrap();
            registers.get_t(return_register as usize)?
        };
        match function {
            Function::Native(f) => {
                let r = f(
                    &native_functions,
                    arguments
                        .to_vec()
                        .iter()
                        .filter(|v| v.is_some())
                        .map(|v| v.unwrap())
                        .collect(),
                )?;
                let mut rc = self.register_stack.borrow_mut();
                let registers = rc.last_mut().unwrap();
                registers.set(return_register as usize, r)?;
            }
            Function::UserDefined(new_i, _nargs, _skip) => {
                let new_i = *new_i;
                self.call_stack.push((self.pc, return_register));
                self.pc = new_i;
                let new_register_set = self.create_new_register_set()?;
                self.register_stack.borrow_mut().push(new_register_set);
            }
        };
        Ok(())
    }

    fn value_to_register(&mut self, register: u8, value: Value) -> Result<(), Error> {
        let mut rc = self.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        match value {
            Value::Constant(v) => {
                registers.set(register as usize, v)?;
            }
            Value::Register(source) => {
                let source_value = registers.get(source as usize)?;
                registers.set(register as usize, source_value)?;
            }
        };
        Ok(())
    }

    fn create_new_register_set(&self) -> Result<RegisterSet, Error> {
        let address = self.allocator.borrow_mut().malloc(256)?;
        Ok(RegisterSet {
            address,
            memory: self.memory.clone(),
            size: 256,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn it_should_add_a_new_function_on_fd() {
        let instructions = vec![
            RevmInstruction::Fd {
                name: "test".to_owned(),
                args: 0,
                skip: 1,
            },
            RevmInstruction::Leave,
        ];
        let program = Program(instructions);
        let mut cpu = CpuRevm::new(1024).unwrap();
        cpu.execute_program(program).unwrap();
        let test = cpu.functions.get("test").unwrap();
        match test {
            Function::UserDefined(ref start, ref args, ref skip) => {
                assert_eq!(*start, 0);
                assert_eq!(*args, 0);
                assert_eq!(*skip, 1);
            }
            _ => panic!("Saved function should be user defined"),
        }
    }

    #[test]
    fn it_should_add_a_constant_to_a_register() {
        let instructions = vec![RevmInstruction::Mov {
            register: 0,
            value: Value::Constant(42),
        }];
        let program = Program(instructions);
        let mut cpu = CpuRevm::new(1024).unwrap();
        cpu.execute_program(program).unwrap();
        let mut rc = cpu.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        assert_eq!(registers.get(0).unwrap(), 42);
    }

    #[test]
    fn it_should_add_a_register_to_a_register() {
        let instructions = vec![RevmInstruction::Mov {
            register: 0,
            value: Value::Register(1),
        }];
        let program = Program(instructions);
        let mut cpu = CpuRevm::new(1024).unwrap();
        {
            let mut rc = cpu.register_stack.borrow_mut();
            let registers = rc.last_mut().unwrap();
            registers.set(1, 42).unwrap();
        }
        cpu.execute_program(program).unwrap();
        let mut rc = cpu.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        assert_eq!(registers.get(0).unwrap(), 42);
    }

    #[test]
    fn it_should_copy_a_global_to_a_register() {
        let instructions = vec![RevmInstruction::Gg {
            string: "test".to_owned(),
            register: 0,
        }];
        let program = Program(instructions);
        let mut cpu = CpuRevm::new(1024).unwrap();
        cpu.globals.insert("test".to_owned(), 42);
        cpu.execute_program(program).unwrap();
        let mut rc = cpu.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        assert_eq!(registers.get(0).unwrap(), 42);
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: GlobalNotFound { name: \"test\" }"
    )]
    fn it_should_panic_when_copying_from_an_unexisting_global() {
        let instructions = vec![RevmInstruction::Gg {
            string: "test".to_owned(),
            register: 0,
        }];
        let program = Program(instructions);
        let mut cpu = CpuRevm::new(1024).unwrap();
        cpu.execute_program(program).unwrap();
    }

    #[test]
    fn it_should_copy_a_register_to_a_global() {
        let instructions = vec![RevmInstruction::Sg {
            string: "test".to_owned(),
            register: 0,
        }];
        let program = Program(instructions);
        let mut cpu = CpuRevm::new(1024).unwrap();
        {
            let mut rc = cpu.register_stack.borrow_mut();
            let registers = rc.last_mut().unwrap();
            registers.set(0, 42).unwrap();
        }
        cpu.execute_program(program).unwrap();
        let global = cpu.globals.get("test").unwrap().clone();
        assert_eq!(global, 42);
    }

    #[test]
    fn it_should_load_a_u8_into_a_register() {
        let instructions = vec![RevmInstruction::Ld8 {
            register: 0,
            value: Value::Constant(42),
        }];
        let program = Program(instructions);
        let mut cpu = CpuRevm::new(1024).unwrap();
        cpu.execute_program(program).unwrap();
        let mut rc = cpu.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        assert_eq!(registers.get(0).unwrap(), 42);
    }

    #[test]
    fn it_should_load_a_u16_into_a_register() {
        let instructions = vec![RevmInstruction::Ld16 {
            register: 0,
            value: Value::Constant(42),
        }];
        let program = Program(instructions);
        let mut cpu = CpuRevm::new(1024).unwrap();
        cpu.execute_program(program).unwrap();
        let mut rc = cpu.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        assert_eq!(registers.get(0).unwrap(), 42);
    }

    #[test]
    fn it_should_load_a_u32_into_a_register() {
        let instructions = vec![RevmInstruction::Ld32 {
            register: 0,
            value: Value::Constant(42),
        }];
        let program = Program(instructions);
        let mut cpu = CpuRevm::new(1024).unwrap();
        cpu.execute_program(program).unwrap();
        let mut rc = cpu.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        assert_eq!(registers.get(0).unwrap(), 42);
    }

    #[test]
    fn it_should_load_a_u64_into_a_register() {
        let instructions = vec![RevmInstruction::Ld64 {
            register: 0,
            value: Value::Constant(42),
        }];
        let program = Program(instructions);
        let mut cpu = CpuRevm::new(1024).unwrap();
        cpu.execute_program(program).unwrap();
        let mut rc = cpu.register_stack.borrow_mut();
        let registers = rc.last_mut().unwrap();
        assert_eq!(registers.get(0).unwrap(), 42);
    }
}
