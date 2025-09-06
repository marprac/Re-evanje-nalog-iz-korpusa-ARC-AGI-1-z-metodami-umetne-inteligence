import json
import dspy
from dspy import LM, configure, ChainOfThought
from typing import List, Dict, Any
import heapq
import math
import copy
import ast
import os
import subprocess
import sys
import argparse

import litellm

LOG_FILE = "log.txt"
FINAL_LOG_FILE = ""#Pot do datoteke za shranjevanje informacij glede uspešnosti reševanja vseh nalog
ALL = 0



def _write_log(level, msg):
    line = f"[{level}] {msg}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

def _write_final_log(level, msg):
    line = f"[{level}] {msg}\n"
    print(line, end="")
    with open(FINAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

def log_info(msg): _write_log("INFO", msg)
def log_ok(msg): _write_log("OK", msg)
def log_warn(msg): _write_log("WARN", msg)
def log_skip(msg): _write_log("SKIP", msg)
def log_final(msg): _write_final_log("FINAL", msg)

def configure_model():
    lm = LM(
        #izberite svoj LLM model
        api_key="",#vstavite svoj api ključ
    )

    configure(
        lm=lm,
    )

    return lm

def load_task(path: str):
    task = json.load(open(path))
    train_pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
    test_inputs = [ex["input"] for ex in task["test"]]
    test_outputs = [ex.get("output") for ex in task["test"]]
    return train_pairs, test_inputs, test_outputs

def pretty_matrix(mat):
    return "\n".join(" ".join(str(x) for x in row) for row in mat)
    

class GetActions(dspy.Signature):
    """
    You have several matrices that represent inputs and outputs of the same complex transformation.
    Provide a list of basic instructions that are applied to the input to get the output.
    Instructions should be general and in natural language so they can be applied to any of the input matrix.
    Include all the input and output matrices in your reasoning.
    Also provide overall context about the transformation, if there are any similarities or clues between pairs of input and output matrices that might help in understanding the transformation,
    include them in the description output.
    Do not provide any specific values in the description output if they are not the same in all cases.
    """

    Matrices: List[Dict[str, List[List[int]]]] = dspy.InputField(description="List of matrices representing the input and output matrices.")
    Instructions: List[str] = dspy.OutputField(description="List of basic instructions that can be applied to the input to get the output.")
    Description: str = dspy.OutputField(description="Description of the transformation.")



def get_actions_from_task(train_pairs):

    generate_actions = ChainOfThought(GetActions)
    out1 = generate_actions(Matrices=train_pairs)
    instructions = out1.Instructions
    task_description = out1.Description

    return instructions, task_description

class ExecuteInstructions(dspy.Signature):
    """
    You have an input matrix. Based on the description of a transformation, follow the instructions given in the field Instructions to transform the input matrix into the result.
    Follow the instructions step by step and apply them to the matrix. Use outputs from previous instruction as inputs to the next instruction. 
    Check if any values were changed that should not be based on the instructions.
    Output only the final matrix.
    """
    
    Description: str = dspy.InputField(description="Description of the transformation.")
    Matrix: List[List[int]] = dspy.InputField(description="Input matrix to which the instructions will be applied.")
    Instructions: List[str] = dspy.InputField(description="Instructions to be applied.")
    Result: List[List[int]] = dspy.OutputField(description="Return ONLY the final matrix as a valid Python list of lists. No explanations, no extra text.")


def apply_actions(input_matrix, actions, description):
    execute_instructions = ChainOfThought(ExecuteInstructions)
    out = execute_instructions(
        Description=description,
        Matrix=input_matrix,
        Instructions=actions
    )
    return out.Result

class ExecuteInstructionsFinal(dspy.Signature):
    """
    You have an input matrix. Based on the description of a transformation, follow the instructions given in the field Instructions to transform the input matrix into the result.
    You also have examples of input and output pairs of the similar transformation, use them as context for your application of the rules.
    Follow the instructions step by step and apply them to the matrix. Use outputs from previous instruction as inputs to the next instruction. 
    Check if any values were changed that should not be based on the instructions.
    Output only the final matrix.
    """
    
    Description: str = dspy.InputField(description="Description of the transformation.")
    Matrix: List[List[int]] = dspy.InputField(description="Input matrix to which the instructions will be applied.")
    Instructions: List[str] = dspy.InputField(description="Instructions to be applied.")
    TrainPairs: List[Dict[str, List[List[int]]]] = dspy.InputField(description="List of example input and output pairs for context.")
    Result: List[List[int]] = dspy.OutputField(description="Return ONLY the final matrix as a valid Python list of lists. No explanations, no extra text.")

def apply_actions_final(input_matrix, actions, description, train_pairs):
    execute_instructions = ChainOfThought(ExecuteInstructionsFinal)
    out = execute_instructions(
        Description=description,
        Matrix=input_matrix,
        TrainPairs=train_pairs,
        Instructions=actions
    )
    return out.Result

class Precond_action(dspy.Signature):
    """
    You have a list of instructions that can be applied to an input matrix to get a desired output matrix.
    Check every instruction and check if it is precondition or action.
    A precondition is an instruction that does not change the matrix but gets some information about the matrix that can be used later.
    An action is an instruction that changes the matrix.
    Return a list of int where 0 means precondition and 1 means action to match the instruction indices.
    """
    
    Instructions: List[str] = dspy.InputField(description="List of instructions that can be applied to the input to get the output.")
    PreconditionOrAction: List[int] = dspy.OutputField(description="Precondition or action that can be applied to the input to get closer to the desired output.")

def precondition_or_action(instructions):
    precond_action = dspy.Predict(Precond_action)
    out = precond_action(Instructions=instructions)
    precond_or_action = out.PreconditionOrAction
    return precond_or_action

def list_all_colors(train_pairs):
    numbers = set()
    for pair in train_pairs:
        for row in pair["input"]:
            numbers.update(row)
        for row in pair["output"]:
            numbers.update(row)
    return sorted(numbers)

def equal(A, B):
    return len(A) == len(B) and len(A[0]) == len(B[0]) and all(
        A[i][j] == B[i][j] for i in range(len(A)) for j in range(len(A[0]))
    )

def diff_coords(A, B):
    diffs = []
    H, W = len(A), len(A[0])
    H2, W2 = len(B), len(B[0])
    for i in range(max(H, H2)):
        for j in range(max(W, W2)):
            a = A[i][j] if i < H and j < W else None
            b = B[i][j] if i < H2 and j < W2 else None
            if a != b:
                diffs.append((i, j))
    return diffs

def matrix_diff_report(A, B, limit=24):
    diffs = diff_coords(A, B)
    rep  = []
    rep.append(f"[DIFF] Razlikujočih celic: {len(diffs)}")
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        rep.append(f"[DIFF] Dimenzije: A={len(A)}x{len(A[0])}, B={len(B)}x{len(B[0])}")
    for (i, j) in diffs[:limit]:
        a = A[i][j] if i < len(A) and j < len(A[0]) else None
        b = B[i][j] if i < len(B) and j < len(B[0]) else None
        rep.append(f"  - ({i},{j}): {a} -> {b}")
    if len(diffs) > limit:
        rep.append(f"  ... in še {len(diffs)-limit} razlik")
    return "\n".join(rep)


class RepairInstruction(dspy.Signature):
    """
    You have a start matrix and a target matrix that represent input and output of a complex transformation.
    You tried to apply an instruction to current matrix but it didn't improve the matrix towards the target.
    Given the transformation description, the start matrix, the current matrix,
    the target matrix, a short diff summary, and the previous instructions, propose a list of instructions that would replace the current instruction and reach the target matrix.
    Rather than providing specific values or positions, focus instructions on how to get those values from the matrix. 
    Do not include any specific values or positions in the instructions just explanation how you got them from the matrix or transformation information.
    Output just the additional instructions that are needed. Do not include any instructions that are already in the previous instructions.
    Keep the instruction in clear, executable natural language. This is very important, instructions should be general, applicable to any input matrix, and not tied to specific values.

    Execute the instructions step by step and apply them to the matrix. Use outputs from previous instruction as inputs to the next instruction.
    Output the final matrix after applying all the instructions.
    """
    Description: str = dspy.InputField(description="Description of the transformation.") 
    StartMatrix: List[List[int]] = dspy.InputField(description="The original input matrix before any instructions were applied.")
    CurrentMatrix: List[List[int]] = dspy.InputField(description="The current state of the input matrix before applying the failed instructions.")
    TargetMatrix: List[List[int]] = dspy.InputField(description="The target output matrix that we want to achieve.")
    PreviousInstructions: List[str] = dspy.InputField(description="A list of instructions that were applied to the start matrix until the failed instruction.")
    FailedInstruction: str = dspy.InputField(description="The instruction that was attempted but failed to improve the matrix.")
    DiffSummary: str = dspy.InputField(description="A summary of the differences between the matrix with failed actions applied and the target matrix.")
    RevisedInstruction: List[str] = dspy.OutputField(description="A only list of instructions that would replace the current instruction and reach the target matrix.")
    FinalMatrix: List[List[int]] = dspy.OutputField(description="Return ONLY the final matrix as a valid Python list of lists. No explanations, no extra text.")


class ProposeNextSteps(dspy.Signature):
    """
    You have a start matrix and a target matrix that represent input and output of a complex transformation.
    You would like to make an instruction list that will transform the start matrix into the target matrix.
    The current matrix is the result of applying some instructions to the start matrix but it is not yet the target matrix.
    Based on the description, the start matrix, the current matrix, the target, and a diff summary,
    propose new next instructions that would transform the current matrix into the target matrix.
    Do not include any specific values or positions in the instructions just explanation how you got them from the matrix or transformation information.
    Make each instruction concise and executable in natural language. 
    Output just the additional instructions that are needed. Do not include any instructions that are already in the current instructions.
    """
    Description: str = dspy.InputField(description="Description of the transformation.")
    StartMatrix: List[List[int]] = dspy.InputField(description="The original input matrix before any instructions were applied.")
    CurrentMatrix: List[List[int]] = dspy.InputField(description="The current state of the input matrix before applying the new instructions.")
    TargetMatrix: List[List[int]] = dspy.InputField(description="The target output matrix that we want to achieve.")
    DiffSummary: str = dspy.InputField(description="A summary of the differences between the current matrix and the target matrix.")
    CurrInstructions: List[str] = dspy.InputField(description="The list of instructions that were applied to the start matrix.")
    NewInstructions: List[str] = dspy.OutputField(description="Return ONLY a JSON list of concise revised natural-language instructions, no extra commentary.")


def find_in_one_pair(input_matrix, output_matrix, description, seq_first, what_kind, pair_idx):

    start_state = input_matrix
    target = output_matrix
    
    def dim_penalty(mat):
        return abs(len(mat) - len(target)) + abs(len(mat[0]) - len(target[0]))

    def heuristic(mat):
        return len(diff_coords(mat, target)) + dim_penalty(mat)
    

    def check_instructions(seq, what_kind):

        new_seq, best_mat, best_h = [], start_state, heuristic(start_state)
        i = 0

        
        work_seq = list(seq)
        what_kind = list(what_kind)
        inst_to_check = len(work_seq)


        while i < inst_to_check:

            if i >= len(work_seq):
                log_warn("Izven obsega navodil, izstopam iz zanke.")
                break
            
            if i >= len(what_kind):
                what_kind = precondition_or_action(work_seq)

            if what_kind[i] == 0:
                new_seq.append(work_seq[i])
                log_info(f"Preverjam navodilo {i}/{inst_to_check}.")
                i += 1
                continue

            cand_seq = new_seq + [work_seq[i]]
            cand_mat= apply_actions(start_state, cand_seq, description)
            cand_h = heuristic(cand_mat)

            if cand_h <= best_h:
                if not (best_h == heuristic(start_state) and cand_h/(len(target)*len(target[0])) > 0.5 and (len(cand_mat) != len(start_state) or len(cand_mat[0]) != len(start_state[0]))):
                    log_info(f"Navodilo {i} je izboljšalo heuristiko.")
                    new_seq, best_mat, best_h = cand_seq, cand_mat, cand_h
                    log_ok(f" Obdržim navodilo {i}: '{work_seq[i]}' | h={best_h}")
                    if equal(best_mat, target):
                        log_ok(f"Rešeno z navodili na koraku {i}.")
                        return new_seq, best_mat
                    i += 1
                    continue
            log_info(f"Navodilo {i} ni izboljšalo heuristike.")
            diff_txt = matrix_diff_report(cand_mat, target)
            fixer = ChainOfThought(RepairInstruction)
            out = fixer(
                Description=description,
                StartMatrix=start_state,
                CurrentMatrix=best_mat,
                TargetMatrix=target,
                PreviousInstructions=new_seq,
                FailedInstruction=work_seq[i],
                DiffSummary=diff_txt
            )
            revised_instrs = out.RevisedInstruction
            final_mat = out.FinalMatrix
            final_h = heuristic(final_mat)

            if final_h < best_h:
                best_h, best_mat = final_h, final_mat
                new_seq = new_seq + revised_instrs
                work_seq = new_seq + work_seq[i+1:] 
                what_kind = precondition_or_action(work_seq)
                inst_to_check = len(work_seq)
                i = len(new_seq)
                log_info("Popravljeno navodilo:")
                for instr in work_seq:
                    log_info(f"- {instr}")
                if equal(best_mat, target):
                    log_ok(f"Train par je rešen s popravljenimi navodili na koraku {i}.")
                    return new_seq, best_mat
                log_info(f"Najboljša matrika po popravilu:\n" + "\n".join(str(row) for row in best_mat))
            else:
                log_skip(f"Preskakujem korak {i}: '{work_seq[i]}' (ni izboljšave po popravilu).")
                i += 1

        return new_seq, best_mat

    def add_instructions(new_seq, best_mat, description, start_state, target):
        log_warn("Navodila niso rešila para, poskušam z iskanjem dodatnih navodil.")
        diff_txt = matrix_diff_report(best_mat, target)
        proposer = ChainOfThought(ProposeNextSteps)
        prop = proposer(
            Description=description,
            StartMatrix=start_state,
            CurrentMatrix=best_mat,
            TargetMatrix=target,
            DiffSummary=diff_txt,
            CurrInstructions=new_seq
        )

        new_instrs = prop.NewInstructions

        extended_seq = new_seq + new_instrs
        extended_kind = precondition_or_action(extended_seq)

        new_seq, best_mat = check_instructions(extended_seq, extended_kind)
        return new_seq, best_mat

    log_info(f"Popravljanje navodil za par {pair_idx}.")
    new_seq, best_mat = check_instructions(seq_first, what_kind)

    log_info("Najboljša navodila:")
    for instr in new_seq:
        log_info(f"- {instr}")
    log_info("Matrika iz najboljših navodil:\n" +
                "\n".join(str(row) for row in best_mat))
    
    seq_old = None
    index = 0

    while not equal(best_mat, target) and len(new_seq) < 16:
        seq_old = list(new_seq)
        new_seq, best_mat = add_instructions(new_seq, best_mat, description, start_state, target)
        index += 1
        log_warn(f"Zanka št. {index}" )

        if new_seq == seq_old:
            log_warn("Navodila se ne spreminjajo, izstopam iz zanke.")
            break

    if equal(best_mat, target):
        log_ok(f"Par {pair_idx} je rešen z navodili:")
        for instr in new_seq:
            log_ok(f"- {instr}")
    else:
        log_warn(f"Par {pair_idx} ni rešen, najboljša matrika:\n" +
                "\n".join(str(row) for row in best_mat))
        log_warn("Ciljna matrika:\n" +
                "\n".join(str(row) for row in target))
        log_warn("Najboljša navodila:")
        for instr in new_seq:
            log_warn(f"- {instr}")

    return new_seq


def find_sequence(train_pairs, seq_first, description, what_kind):
    global ALL

    seq = seq_first
    what_kind = precondition_or_action(seq)

    for idx,pair in enumerate(train_pairs[0:], start=1):
        test = apply_actions(pair["input"], seq, description)
        log_info("#####################################################################################")

        if not equal(test, pair["output"]):
            log_info(f"Pair {idx} ni rešen, dodajam korekcijo...")
            log_info(f"Moj output:\n" +
                "\n".join(str(row) for row in test) +
                "\n\nTest output:\n" +
                "\n".join(str(row) for row in pair["output"])
            )
        else:
            log_ok(f"Pair {idx} je rešen z dosedanjimi navodili.")
            continue

        seq = find_in_one_pair(pair["input"], pair["output"], description, seq, what_kind, pair_idx=idx)
        what_kind = precondition_or_action(seq)


    return seq, what_kind


def find_output(task_path):
    global ALL
    train_pairs, test_inputs, test_outputs = load_task(task_path)
    train_pairs = [{"input": inp, "output": out} for inp, out in train_pairs]

    log_info("Iskanje začetnih akcij iz naloge.")
    seq_first, task_description = get_actions_from_task(train_pairs)
    prec_first = precondition_or_action(seq_first)
    log_info(f"Začetni opis naloge: {task_description}")
    log_info("Začetna navodila:")
    for instr in seq_first:
        log_info(f"- {instr}")
    log_info(f"Predpogoji ali akcija: {prec_first}")

    log_info("Testiranje začetnih akcij na učnih primerih.")
    seq, what_kind = find_sequence(train_pairs, seq_first, description=task_description, what_kind=prec_first)

    log_info("Naučena navodila:")
    for instr in seq:
        log_info(f"- {instr}")
    log_info(f"Predpogoji ali akcija: {what_kind}")

    idx = 0
    wrong = 0
    for inp, out in zip(test_inputs, test_outputs):
        idx += 1

        test = apply_actions_final(inp, seq, task_description, train_pairs)

        if not equal(test, out):
            wrong += 1
            log_warn("#####################################################################################")
            log_warn(f"Test primer {idx} ni pravilno rešen.")
            log_warn(matrix_diff_report(test, out))
            log_warn("Matrika iz naučenih navodil:\n" + "\n".join(str(row) for row in test))
            log_warn("Pravilna matrika:\n" + "\n".join(str(row) for row in out))
        else:
            log_ok(f"Test primer {idx} je pravilno rešen.")

        if idx > 5 and wrong/idx > 0.5:
            log_warn("Preveč napačnih test primerov, ustavljam preverjanje.")
            break


 
    base_name = os.path.splitext(os.path.basename(task_path))[0]

    if wrong == 0:
        log_ok("Vsi testi so bili uspešno rešeni.")
        log_final("Naloga " + base_name + ": OK")
        ALL += 1
    else:  
        log_warn(f"Napaka pri reševanju naloge. {wrong} od {idx} testov ni bilo pravilno rešenih.")
        log_final("Naloga " + base_name + ": NAPAKA")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Pot do ene JSON naloge (worker način)")
    args = parser.parse_args()

    if args.task:
        lm = configure_model()

        task_path = args.task
        base_name = os.path.splitext(os.path.basename(task_path))[0]
        log_dir = ""#Pot do datoteke za shranjevanje informacij reševanja posamezne naloge
        os.makedirs(log_dir, exist_ok=True)
        LOG_FILE = os.path.join(log_dir, base_name + ".txt")

        log_info(f"===============================================")
        log_info(f"Začel iskanje za nalogo {os.path.basename(task_path)}.")

        try:
            find_output(task_path)
        except Exception as e:
            log_warn(f"Napaka pri obdelavi {os.path.basename(task_path)}: {e}")

        log_info(f"Iskanje za {os.path.basename(task_path)} zaključeno.")
        log_info(f"===============================================\n")
        sys.exit(0)

    training_dir = ""#Pot do mape z JSON nalogami za reševanje
    log_dir = ""#Pot do mape za shranjevanje informacij reševanja posamezne naloge
    os.makedirs(log_dir, exist_ok=True)

    for file in os.listdir(training_dir):
        if not file.endswith(".json"):
            continue

        task_path = os.path.join(training_dir, file)

        try:
            subprocess.run([sys.executable, __file__, "--task", task_path], check=False)
        except Exception as e:
            base_name = os.path.splitext(file)[0]
            LOG_FILE = os.path.join(log_dir, base_name + ".txt")
            log_warn(f"Podproces za {file} je padel: {e}")

    ok = 0
    try:
        with open(FINAL_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if " OK" in line or ": OK" in line:
                    ok += 1
    except FileNotFoundError:
        pass
    log_final(f"Skupaj rešenih: {ok}/400")
    log_info("Celoten trening zaključen.")
