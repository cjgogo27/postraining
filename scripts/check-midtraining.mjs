import {readdir,readFile} from 'node:fs/promises';
import path from 'node:path';
const root=path.resolve('midtraining/src/content/docs'),base='/postraining/midtraining/';
async function walk(dir){return(await Promise.all((await readdir(dir,{withFileTypes:true})).map(x=>x.isDirectory()?walk(path.join(dir,x.name)):path.join(dir,x.name)))).flat();}
const files=(await walk(root)).filter(x=>/\.mdx?$/.test(x)),routes=new Set([base,'/postraining/']);
for(const file of files)routes.add(base+path.relative(root,file).replaceAll('\\','/').replace(/\.mdx?$/,'').replace(/^index$/,'')+'/');
for(const file of await walk('midtraining/public'))routes.add(base+path.relative('midtraining/public',file).replaceAll('\\','/'));
const chapters=JSON.parse(await readFile('midtraining/src/data/chapters.json','utf8'));
const qa=JSON.parse(await readFile('midtraining/src/data/section-qa.json','utf8'));
const errors=[];let chinese=0;
for(const file of files){
 const text=await readFile(file,'utf8'),relative=path.relative(root,file).replaceAll('\\','/');
 chinese+=(text.match(/[\u3400-\u9fff]/g)||[]).length;
 if(!/^---\r?\ntitle:/m.test(text))errors.push(relative+': missing title');
 if(/TODO|TBD|待补充|PLACEHOLDER/.test(text))errors.push(relative+': unfinished text');
 for(const m of text.matchAll(/(?:\]\(|(?:href|link)=['"])(\/postraining\/[^)'"\s]*)/g))if(!routes.has(m[1].split(/[?#]/)[0]))errors.push(relative+': unknown route '+m[1]);
 if(/^0[1-6]-/.test(relative)){
  const entry=chapters.find(x=>relative===x.path+'.mdx');if(!entry){errors.push(relative+': not indexed');continue;}
  const id=entry.title.split(' ')[0];
  if(qa[id]?.length!==3||qa[id].some(x=>!x.q||!x.a||!x.role))errors.push(id+': role QA');
  for(const mark of ['系统位置','一句话听懂','到这里，你应该已经能听懂','## 参考资料',`id="${id}" mode="ask"`,`id="${id}" mode="answer"`])if(!text.includes(mark))errors.push(relative+': missing '+mark);
  if((text.match(/\[\d+\] \[/g)||[]).length<2)errors.push(relative+': expected primary reading sources');
 }
}
if(chapters.length!==29||Object.keys(qa).length!==29)errors.push('expected 29 sections/QA groups');
if(errors.length){console.error(errors.join('\n'));process.exit(1);}
console.log(JSON.stringify({status:'PASS',book:'midtraining',pages:files.length,sections:29,role_questions:87,chinese_characters:chinese}));
