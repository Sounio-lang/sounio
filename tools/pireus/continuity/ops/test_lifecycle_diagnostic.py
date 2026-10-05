import ast
import json
from pathlib import Path
import tempfile
import time
import unittest
from build_lifecycle_diagnostic import BASE, generate
from lifecycle_observer import Observer, fields

class Cuda:
    def memory_allocated(self): return 100
    def memory_reserved(self): return 200
    def max_memory_allocated(self): return 150
    def max_memory_reserved(self): return 250

class Tests(unittest.TestCase):
    def test_missing_is_null(self):
        values,error=fields("/definitely-absent-pireus-file",("Rss",))
        self.assertEqual(values,{"Rss":None})
        self.assertIsNotNone(error)
    def test_observer_lifecycle_and_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"events.jsonl"
            o=Observer(p,"42","1",Cuda(),interval=0.1)
            o.start()
            o.mark("DECODE_ENTRY",3,0)
            time.sleep(0.15)
            o.mark("CLEANUP_AFTER",3,121)
            o.close()
            self.assertFalse(o.thread.is_alive())
            rows=[json.loads(x) for x in p.read_text().splitlines()]
            self.assertTrue(any(x["stage"]=="HOST_SAMPLE" for x in rows))
            self.assertEqual(rows[-1]["stage"],"OBSERVER_END")
            event=next(x for x in rows if x["stage"]=="DECODE_ENTRY")
            self.assertEqual(event["cuda"],dict(allocated=100,reserved=200,peak_allocated=150,peak_reserved=250))
            self.assertEqual((event["job"],event["rank"],event["index"]),("42","1",3))
            self.assertTrue(all(x["observation_duration_ns"]>=0 for x in rows))
            with self.assertRaises(FileExistsError): Observer(p,"42","1",Cuda())
    def test_bad_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"events"
            with self.assertRaises(ValueError): Observer(p,"1","0",Cuda(),interval=0)
            self.assertFalse(p.exists())
    def test_base_mutation_refused(self):
        with self.assertRaises(ValueError):
            generate((BASE/"runtime/offline_generate.py").read_bytes()+b" ", "")
    def test_algorithm_ast_unchanged(self):
        raw=(BASE/"runtime/offline_generate.py").read_bytes()
        observer=Path(__file__).with_name("lifecycle_observer.py").read_text()
        tree=ast.parse(generate(raw,observer))
        class Strip(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call):
                    fn=node.value.func
                    if isinstance(fn,ast.Attribute) and isinstance(fn.value,ast.Name) and fn.value.id in ("lifecycle","atexit"):
                        return None
                    if isinstance(fn,ast.Name) and fn.id=="exec" and "_lifecycle_namespace" in ast.unparse(node):
                        return None
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if any(isinstance(x,ast.Name) and x.id in ("lifecycle","_lifecycle_namespace") for x in node.targets): return None
                return self.generic_visit(node)
            def visit_Import(self,node):
                if [x.name for x in node.names]==["atexit"]: return None
                return node
            def visit_If(self,node):
                node=self.generic_visit(node)
                if not node.body: return None
                return node
        stripped=Strip().visit(tree)
        self.assertEqual(ast.dump(stripped,include_attributes=False),
                         ast.dump(ast.parse(raw),include_attributes=False))
if __name__=="__main__": unittest.main()
