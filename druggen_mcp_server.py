from mcp.server.fastmcp import FastMCP
import pandas as pd
import ast
from DrugGen.drugGen_generator import run_inference
import json
from tempfile import TemporaryDirectory
from pathlib import Path

mcp = FastMCP("DrugGen")

@mcp.tool(name="run_druggen")
def run_druggen(uniprot_id: str, num_generated: int = 7) -> str:
    try:
        if not uniprot_id.strip() or not 1 <= num_generated <= 100:
            raise ValueError("Provide a UniProt ID and num_generated between 1 and 100")
        with TemporaryDirectory(prefix="drugforge-") as folder:
            output_file = str(Path(folder) / "output.txt")
            run_inference(uniprot_ids=[uniprot_id], num_generated=num_generated, output_file=output_file)
            df = pd.read_csv(output_file, sep='\t')
        smiles_raw_list = df['SMILES'].tolist()
        smiles_list = []
        for smiles_str in smiles_raw_list:
            parsed = ast.literal_eval(smiles_str)
            smiles_list.extend(parsed)
        if not smiles_list:
            raise RuntimeError("DrugGen returned no candidate molecules")
        return json.dumps({"smiles": smiles_list[:num_generated]})
    except Exception as e:
        return json.dumps({"error": str(e), "smiles": []})

if __name__ == "__main__":
    mcp.run(transport="stdio")