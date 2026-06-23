import os
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Define Medical Knowledge Base (12 Target Classes + 1 sub-class)
medical_kb = [
    {
        "id": "NSR",
        "condition": "Normal Sinus Rhythm",
        "definition": "The normal, regular rhythm of the heart set by the natural pacemaker of the heart called the sinoatrial (SA) node.",
        "ecg_characteristics": "P wave precedes every QRS complex. Constant PR interval. Normal P wave axis.",
        "diagnostic_criteria": "Heart rate between 60-100 bpm. Normal PR interval (0.12-0.20s). Normal QRS duration (<0.12s).",
        "typical_hr_range": "60 to 100 bpm",
        "clinical_significance": "Indicates a healthy, functioning electrical conduction system in the heart.",
        "common_causes": "Normal physiology.",
        "treatment": "None required.",
        "references": "AHA ECG Guidelines."
    },
    {
        "id": "Sinus_Tachycardia",
        "condition": "Sinus Tachycardia",
        "definition": "A fast heartbeat that originates from the SA node.",
        "ecg_characteristics": "Normal P waves preceding every QRS. Very fast, regular rhythm. Shorter RR intervals.",
        "diagnostic_criteria": "Heart rate > 100 bpm. Normal PR and QRS.",
        "typical_hr_range": "> 100 bpm (usually 100-160 bpm)",
        "clinical_significance": "Often a physiological response to stress, exercise, fever, or pain, but can indicate underlying pathology.",
        "common_causes": "Exercise, anxiety, fever, anemia, hyperthyroidism.",
        "treatment": "Treat underlying cause. Beta-blockers if symptomatic.",
        "references": "AHA Arrhythmia Guidelines."
    },
    {
        "id": "Sinus_Arrhythmia",
        "condition": "Sinus Arrhythmia",
        "definition": "A normal variation in heart rate that changes with the breathing cycle.",
        "ecg_characteristics": "Irregular rhythm with normal P waves. Heart rate increases during inspiration and decreases during expiration.",
        "diagnostic_criteria": "P-P interval varies by more than 0.16 seconds. Normal PR and QRS.",
        "typical_hr_range": "60 to 100 bpm, but variable.",
        "clinical_significance": "Usually a sign of good cardiovascular health and high vagal tone. Common in young, healthy individuals.",
        "common_causes": "Respiration (Respiratory sinus arrhythmia).",
        "treatment": "No treatment needed.",
        "references": "ACC/AHA Clinical Practice Guidelines."
    },
    {
        "id": "PAC",
        "condition": "Premature Atrial Contraction",
        "definition": "Early beats originating in the atria outside of the SA node.",
        "ecg_characteristics": "Premature, abnormally shaped P wave followed by a normal narrow QRS complex. Followed by an incomplete compensatory pause.",
        "diagnostic_criteria": "Premature P wave with a different morphology than the sinus P wave. P-R interval may be prolonged.",
        "typical_hr_range": "Variable. Background rhythm usually normal.",
        "clinical_significance": "Usually benign. Frequent PACs may trigger atrial fibrillation or atrial flutter.",
        "common_causes": "Stress, fatigue, caffeine, alcohol, tobacco, electrolyte imbalances.",
        "treatment": "Often observation. Lifestyle modification. Beta-blockers if highly symptomatic.",
        "references": "ESC Guidelines on Supraventricular Arrhythmias."
    },
    {
        "id": "RBBB",
        "condition": "Right Bundle Branch Block",
        "definition": "A delay or blockage of electrical impulses to the right ventricle.",
        "ecg_characteristics": "Widened QRS. RSR' pattern ('bunny ears') in leads V1-V2. Wide, slurred S wave in leads I and V6.",
        "diagnostic_criteria": "QRS duration >= 0.12s. RSR' in right precordial leads.",
        "typical_hr_range": "Variable. Does not affect heart rate directly.",
        "clinical_significance": "Can be normal in healthy individuals, but may indicate right ventricular strain, pulmonary embolism, or structural heart disease.",
        "common_causes": "Cor pulmonale, pulmonary embolism, congenital heart disease, normal variant.",
        "treatment": "Treat underlying condition. RBBB alone usually requires no specific treatment.",
        "references": "AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram."
    },
    {
        "id": "LBBB",
        "condition": "Left Bundle Branch Block",
        "definition": "A delay or blockage of electrical impulses to the left ventricle.",
        "ecg_characteristics": "Widened QRS. Broad, notched, or slurred R wave in leads I, aVL, V5, V6. Absence of septal Q waves.",
        "diagnostic_criteria": "QRS duration >= 0.12s. Broad monophasic R waves in lateral leads.",
        "typical_hr_range": "Variable.",
        "clinical_significance": "Almost always indicates underlying structural heart disease. Masks ECG signs of acute myocardial infarction.",
        "common_causes": "Ischemic heart disease, hypertension, aortic stenosis, dilated cardiomyopathy.",
        "treatment": "Treat underlying disease. May require cardiac resynchronization therapy (CRT) if heart failure is present.",
        "references": "AHA/ACCF/HRS Recommendations."
    },
    {
        "id": "IVCD",
        "condition": "Intraventricular Conduction Delay",
        "definition": "A generic term for widening of the QRS complex that does not meet strict criteria for LBBB or RBBB.",
        "ecg_characteristics": "Widened QRS (> 0.11s) without classic LBBB/RBBB morphology.",
        "diagnostic_criteria": "QRS duration > 0.11s. Atypical morphology.",
        "typical_hr_range": "Variable.",
        "clinical_significance": "Associated with fibrosis, electrolyte abnormalities, or drug toxicity. Increased risk of mortality in heart failure.",
        "common_causes": "Hyperkalemia, antiarrhythmic drugs, myocardial fibrosis.",
        "treatment": "Dependent on underlying cause.",
        "references": "AHA/ACCF/HRS Recommendations."
    },
    {
        "id": "Delta_Wave",
        "condition": "Delta Wave (WPW Pattern)",
        "definition": "An early, slurred upstroke of the QRS complex due to pre-excitation via an accessory pathway (Bundle of Kent).",
        "ecg_characteristics": "Short PR interval (< 0.12s) followed by a slurred upstroke (delta wave) widening the QRS.",
        "diagnostic_criteria": "PR interval < 0.12s, Delta wave present, QRS > 0.11s.",
        "typical_hr_range": "Variable. Can trigger extreme tachycardias (AVRT).",
        "clinical_significance": "Indicates Wolff-Parkinson-White syndrome if accompanied by tachyarrhythmias. High risk of sudden cardiac death if combined with atrial fibrillation.",
        "common_causes": "Congenital accessory electrical pathway.",
        "treatment": "Catheter ablation of the accessory pathway if symptomatic.",
        "references": "ESC Guidelines on Supraventricular Arrhythmias."
    },
    {
        "id": "Persistent_ST_Elevation",
        "condition": "Persistent ST Elevation",
        "definition": "Elevation of the ST segment above the baseline, indicating acute or prior myocardial injury.",
        "ecg_characteristics": "ST segment is abnormally high in continuous leads.",
        "diagnostic_criteria": "ST elevation > 1mm in limb leads or > 2mm in precordial leads.",
        "typical_hr_range": "Variable.",
        "clinical_significance": "Medical emergency. Usually indicates an acute transmural myocardial infarction (STEMI) or pericarditis.",
        "common_causes": "Myocardial infarction, pericarditis, ventricular aneurysm, early repolarization.",
        "treatment": "Immediate reperfusion therapy (PCI or thrombolytics) if STEMI.",
        "references": "AHA/ACC STEMI Guidelines."
    },
    {
        "id": "LAE",
        "condition": "Left Atrial Enlargement",
        "definition": "Hypertrophy or dilation of the left atrium.",
        "ecg_characteristics": "Broad, notched P waves (P mitrale) in lead II (> 0.12s). Deep negative terminal component of P wave in lead V1.",
        "diagnostic_criteria": "P wave duration > 0.12s in lead II. Terminal negative P wave in V1 > 1mm deep and > 0.04s wide.",
        "typical_hr_range": "Variable.",
        "clinical_significance": "Marker of chronically elevated left atrial pressure. Increases risk of atrial fibrillation.",
        "common_causes": "Mitral valve disease, hypertension, left ventricular hypertrophy.",
        "treatment": "Manage underlying cause (blood pressure, valvular disease).",
        "references": "AHA/ACCF/HRS Recommendations."
    },
    {
        "id": "VFib",
        "condition": "Ventricular Fibrillation",
        "definition": "A chaotic, disorganized electrical rhythm originating in the ventricles, causing the heart to quiver instead of pump.",
        "ecg_characteristics": "Irregular, chaotic, bizarre waves of varying amplitude and shape. No identifiable P waves, QRS complexes, or T waves.",
        "diagnostic_criteria": "No identifiable waveforms. Rapid, chaotic baseline.",
        "typical_hr_range": "Cannot be determined. 0 effective mechanical bpm.",
        "clinical_significance": "Cardiac arrest. Fatal within minutes if untreated.",
        "common_causes": "Acute myocardial infarction, ischemia, electrolyte disturbances, severe heart failure.",
        "treatment": "Immediate CPR and defibrillation.",
        "references": "AHA Guidelines for CPR and ECC."
    },
    {
        "id": "VFlutter",
        "condition": "Ventricular Flutter",
        "definition": "An extremely rapid ventricular tachycardia with a sinusoidal waveform.",
        "ecg_characteristics": "Continuous sine wave appearance. No distinction between QRS and T waves.",
        "diagnostic_criteria": "Rate > 200 bpm. Sine-wave morphology.",
        "typical_hr_range": "200 to 300 bpm.",
        "clinical_significance": "Hemodynamically unstable. Often deteriorates rapidly into ventricular fibrillation.",
        "common_causes": "Severe ischemia, drug toxicity.",
        "treatment": "Immediate cardioversion or defibrillation.",
        "references": "AHA Guidelines for CPR and ECC."
    },
    {
        "id": "Pacemaker",
        "condition": "Pacemaker Rhythm",
        "definition": "A rhythm driven by an artificial electronic pacemaker device.",
        "ecg_characteristics": "Sharp, vertical pacing spikes preceding the P wave (atrial pacing) or QRS complex (ventricular pacing).",
        "diagnostic_criteria": "Presence of pacemaker spikes.",
        "typical_hr_range": "Usually fixed at a set rate (e.g., 60-70 bpm) or tracks atrial activity.",
        "clinical_significance": "Indicates patient has an implanted device. Must verify capture (spike is followed by a complex) and sensing.",
        "common_causes": "Sick sinus syndrome, high-degree AV block.",
        "treatment": "Routine device interrogation and monitoring.",
        "references": "HRS/EHRA Expert Consensus on Pacemaker Management."
    }
]

def build_kb():
    print("Building CardioVision Medical Knowledge Base in ChromaDB...")
    
    # 2. Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./backend/chromadb")
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(name="cardiovision_kb")
    
    # 3. Load embedding model
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Prepare documents
    documents = []
    ids = []
    metadatas = []
    embeddings = []
    
    for entry in medical_kb:
        # Create a rich text representation of the medical knowledge for the LLM
        doc_text = f"Condition: {entry['condition']}\n"
        doc_text += f"Definition: {entry['definition']}\n"
        doc_text += f"Diagnostic Criteria: {entry['diagnostic_criteria']}\n"
        doc_text += f"ECG Characteristics: {entry['ecg_characteristics']}\n"
        doc_text += f"Typical Heart Rate: {entry['typical_hr_range']}\n"
        doc_text += f"Clinical Significance: {entry['clinical_significance']}\n"
        doc_text += f"Common Causes: {entry['common_causes']}\n"
        doc_text += f"Treatment: {entry['treatment']}\n"
        doc_text += f"References: {entry['references']}"
        
        documents.append(doc_text)
        ids.append(entry['id'])
        metadatas.append({"condition": entry['condition']})
        
        # Embed
        emb = embedder.encode(doc_text).tolist()
        embeddings.append(emb)
        
    # 5. Insert into Chroma
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Successfully ingested {len(medical_kb)} clinical documents into ChromaDB.")

if __name__ == "__main__":
    # Ensure backend dirs exist
    os.makedirs("./backend", exist_ok=True)
    os.makedirs("./backend/rag", exist_ok=True)
    build_kb()
