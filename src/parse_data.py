import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# Resolve root directory relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------------------------
# Pydantic Schemas for Data Validation
# ---------------------------------------------------------------------------

class PrimaryContact(BaseModel):
    name: str
    title: str

class AccountSchema(BaseModel):
    account_id: str
    company: str
    tam: str
    plan_tier: str
    arr_usd: int
    seats_licensed: int
    seats_active: int
    products: List[str]
    health_status: str
    usage_trend: str
    open_tickets: int
    p1_tickets_last_30d: int
    customer_since: str
    renewal_date: str
    last_qbr_date: str
    primary_contact: PrimaryContact
    escalation_notes: List[str] = Field(default_factory=list)
    nps_score: Optional[int] = None
    last_login_days_ago: int
    integrations_active: List[str] = Field(default_factory=list)
    region: str
    industry: str

class TicketSchema(BaseModel):
    ticket_id: str
    account_id: str
    product: str
    priority: Optional[str] = "P3"
    status: str
    subject: str
    description: Optional[str] = ""
    created_at: str
    updated_at: Optional[str] = None
    satisfaction_score: Optional[int] = None

# ---------------------------------------------------------------------------
# Data Loading and Validation Helpers
# ---------------------------------------------------------------------------

def load_accounts(data_dir: str = DEFAULT_DATA_DIR) -> List[Dict[str, Any]]:
    """Loads and validates accounts.json into a list of account record dicts."""
    file_path = os.path.join(data_dir, "accounts.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Accounts file not found at: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    validated_records = []
    for idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            continue
        try:
            acc = AccountSchema(**item)
            validated_records.append(acc.model_dump())
        except ValidationError as e:
            print(f"[Warning] Account record at index {idx} failed validation: {e}")
            
    return validated_records

def load_tickets(data_dir: str = DEFAULT_DATA_DIR) -> List[Dict[str, Any]]:
    """Loads and validates all tickets_batch_*.json files into a list of ticket record dicts."""
    search_pattern = os.path.join(data_dir, "tickets_batch_*.json")
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        print(f"[Warning] No ticket files found matching pattern: {search_pattern}")
        return []
    
    all_tickets = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        for idx, item in enumerate(raw_data):
            try:
                tkt = TicketSchema(**item)
                all_tickets.append(tkt.model_dump())
            except ValidationError as e:
                print(f"[Warning] Ticket record in {path} at index {idx} failed validation: {e}")
                
    return all_tickets

def load_kb(data_dir: str = DEFAULT_DATA_DIR) -> List[Dict[str, str]]:
    """Loads all markdown document files from data_dir into a list of dicts."""
    search_pattern = os.path.join(data_dir, "*.md")
    file_paths = glob.glob(search_pattern)
    
    kb_docs = []
    for path in file_paths:
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        kb_docs.append({
            "doc_id": filename,
            "filename": filename,
            "path": path,
            "content": content
        })
    return kb_docs

if __name__ == "__main__":
    print("=== Testing Data Loading Utility ===")
    accounts = load_accounts()
    print(f"Loaded Accounts: {len(accounts)} records")
    tickets = load_tickets()
    print(f"Loaded Tickets: {len(tickets)} records")
    kb_articles = load_kb()
    print(f"Loaded KB Articles: {len(kb_articles)} documents")