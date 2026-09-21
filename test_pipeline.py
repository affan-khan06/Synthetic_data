import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from core.data import load_source,split,validate,COLUMNS,CATEGORICAL,CATEGORIES,NUMERIC,BILLS,export_profiles
from core.models import Codec
from core.pipeline import load,load_bundle
from core.models import generate
ROOT=Path(__file__).resolve().parent
class RealDataTests(unittest.TestCase):
    def test_provenance_and_real_schema(self):
        df=load_source()
        self.assertEqual(df.shape,(30000,25))
        self.assertEqual(int(df['default'].astype(int).sum()),6636)
        self.assertTrue((df[BILLS]<0).any().any())
        self.assertNotIn('income_m01',df)
    def test_no_partition_overlap(self):
        parts=split(load_source(),42)
        ids=[set(x.ID) for x in parts]
        rows=[set(map(tuple,x[COLUMNS].to_numpy())) for x in parts]
        for i,j in [(0,1),(0,2),(1,2)]:
            self.assertFalse(ids[i]&ids[j]);self.assertFalse(rows[i]&rows[j])
        self.assertEqual(sum(map(len,parts)),29965)
    def test_signed_numeric_codec(self):
        tr,_,_=split(load_source(),42)
        codec=Codec().fit(tr)
        x,c=codec.encode(tr)
        recovered=1000*np.sinh(x.numpy()*codec.scale+codec.mean)
        np.testing.assert_allclose(recovered,tr[NUMERIC].to_numpy(),rtol=1e-5,atol=.1)
        self.assertTrue(np.isfinite(x.numpy()).all())
    def test_generated_profiles(self):
        bundle=load_bundle(ROOT/'demo_run')
        df=generate(bundle,300,123)
        self.assertEqual(df.shape,(300,24))
        self.assertTrue(np.isfinite(df[NUMERIC]).all().all())
        for c in CATEGORICAL:self.assertTrue(df[c].isin(CATEGORIES[c]).all())
        out=export_profiles(df,123)
        self.assertTrue(out.synthetic_id.is_unique)
        self.assertNotIn('ID',out)
        self.assertTrue((out.AGE==out.AGE.round()).all())
    def test_reject_wrong_schema(self):
        df=load_source().head(500)
        with self.assertRaises(ValueError): validate(df.drop(columns='AGE'))
        bad=df.copy();bad['AGE']=-1
        with self.assertRaises(ValueError): validate(bad)
if __name__=='__main__':unittest.main()
