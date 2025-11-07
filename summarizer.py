import difflib
from pdf_utils import pdf_text
from qa_engine import get_llm

def compute_unified_diff(old_txt, new_txt):
    old_lines = old_txt.splitlines()
    new_lines = new_txt.splitlines()
    diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
    return "\n".join(list(diff))

def summarize_diff(old_file, new_file):
    old = pdf_text(old_file)
    new = pdf_text(new_file)
    diff = compute_unified_diff(old, new)
    llm = get_llm()
    prompt = f"Summarize the important policy changes between versions. Focus on high-impact items.\n\n{diff[:5000]}"
    return llm.invoke(prompt).content

def generate_email_summary(policy_section_text, audience="employee"):
    llm = get_llm()
    prompt = (
        f"You are drafting an email to a {audience}. "
        "Write a short email that explains the policy section below, with 3 action items and subject line.\n\n"
        f"Policy excerpt:\n{policy_section_text[:2000]}"
    )
    return llm.invoke(prompt).content
