import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# Vérifier si le GPU est disponible et définir le device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Afficher le device actuel
print(f"Utilisation du device : {device}")
print(torch.cuda.device_count())
# Déplacer le modèle vers le device approprié

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",device_map="auto")
    
# Lire et préparer les données
chemin_fichier = 'niagara.data'  # Mettez à jour avec le chemin de votre fichier
donnees = pd.read_csv(chemin_fichier, sep=';', header=None, names=['Code', 'Description', 'Lien', 'Detail', 'Flag'], on_bad_lines='skip')
donnees['input'] = "C'est quoi le code " + donnees['Code'] + " ?"
donnees['output'] = donnees['Description'] + ". " + donnees['Detail']

donnees['input'] = [str(text) for text in donnees['input']]
donnees['output'] = [str(text) for text in donnees['output']]

# Créer un Dataset Hugging Face
dataset = Dataset.from_pandas(donnees[['input', 'output']])
train_test_split = dataset.train_test_split(test_size=0.1)  # Séparer les données en set d'entraînement et de test
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Fonction de tokenisation
def tokenize_function(examples):
    # Tokeniser les entrées. Ceci crée les input_ids et attention_mask automatiquement.
    model_inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=256)
    
    # Tokeniser également les outputs pour créer les labels.
    # Notez que pour les modèles de type Causal Language Model (comme GPT),
    # les labels sont généralement les mêmes que les input_ids du texte cible (output),
    # sauf que vous pourriez vouloir ignorer le calcul de la perte sur les tokens de padding.
    labels = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=256)["input_ids"]
    
    # Assigner les labels tokenisés aux inputs du modèle.
    model_inputs["labels"] = labels
    
    return model_inputs





# Appliquer la tokenisation
tokenized_train_dataset = dataset.map(tokenize_function, batched=True, batch_size=16)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=16)

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./results_deep",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,    
    weight_decay=0.01,
    fp16=True
)

# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,

)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle entraîné localement
model.save_pretrained("./trained_model_deep")
tokenizer.save_pretrained("./trained_model_deep")

# Tester le modèle avec un exemple du set de test
for i in range(5):  # Tester sur 5 exemples
    inputs = tokenizer(train_dataset[i]['input'], return_tensors="pt", padding=True, truncation=True, max_length=256)
    outputs = model.generate(**inputs)
    print(f"Input: {train_dataset[i]['input']}")
    print(f"Expected Output: {train_dataset[i]['output']}")
    print(f"Model Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")


